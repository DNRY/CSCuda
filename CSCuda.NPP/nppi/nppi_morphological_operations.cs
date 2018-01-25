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
        /// Single-channel 8-bit unsigned integer dilation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three-channel 8-bit unsigned integer dilation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four-channel 8-bit unsigned integer dilation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four-channel 8-bit unsigned integer dilation, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single-channel 16-bit unsigned integer dilation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three-channel 16-bit unsigned integer dilation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four-channel 16-bit unsigned integer dilation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four-channel 16-bit unsigned integer dilation, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single-channel 32-bit floating-point dilation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three-channel 32-bit floating-point dilation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four-channel 32-bit floating-point dilation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four-channel 32-bit floating-point dilation, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single-channel 8-bit unsigned integer dilation with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilateBorder_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three-channel 8-bit unsigned integer dilation with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilateBorder_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 8-bit unsigned integer dilation with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilateBorder_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 8-bit unsigned integer dilation with border control, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilateBorder_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single-channel 16-bit unsigned integer dilation with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilateBorder_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three-channel 16-bit unsigned integer dilation with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilateBorder_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 16-bit unsigned integer dilation with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilateBorder_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 16-bit unsigned integer dilation with border control, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilateBorder_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single-channel 32-bit floating-point dilation with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilateBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three-channel 32-bit floating-point dilation with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilateBorder_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 32-bit floating-point dilation with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilateBorder_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 32-bit floating-point dilation with border control, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilateBorder_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single-channel 8-bit unsigned integer 3x3 dilation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel 8-bit unsigned integer 3x3 dilation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 8-bit unsigned integer 3x3 dilation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 8-bit unsigned integer 3x3 dilation, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single-channel 16-bit unsigned integer 3x3 dilation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel 16-bit unsigned integer 3x3 dilation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 16-bit unsigned integer 3x3 dilation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 16-bit unsigned integer 3x3 dilation, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single-channel 32-bit floating-point 3x3 dilation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel 32-bit floating-point 3x3 dilation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 32-bit floating-point 3x3 dilation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 32-bit floating-point 3x3 dilation, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single-channel 64-bit floating-point 3x3 dilation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3_64f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single-channel 8-bit unsigned integer 3x3 dilation with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3Border_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three-channel 8-bit unsigned integer 3x3 dilation with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3Border_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 8-bit unsigned integer 3x3 dilation with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3Border_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 8-bit unsigned integer 3x3 dilation with border control, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3Border_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single-channel 16-bit unsigned integer 3x3 dilation with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3Border_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three-channel 16-bit unsigned integer 3x3 dilation with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3Border_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 16-bit unsigned integer 3x3 dilation with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3Border_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 16-bit unsigned integer 3x3 dilation with border control, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3Border_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single-channel 32-bit floating-point 3x3 dilation with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3Border_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three-channel 32-bit floating-point 3x3 dilation with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3Border_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 32-bit floating-point 3x3 dilation with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3Border_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 32-bit floating-point 3x3 dilation with border control, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiDilate3x3Border_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single-channel 8-bit unsigned integer erosion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three-channel 8-bit unsigned integer erosion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four-channel 8-bit unsigned integer erosion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four-channel 8-bit unsigned integer erosion, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single-channel 16-bit unsigned integer erosion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three-channel 16-bit unsigned integer erosion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four-channel 16-bit unsigned integer erosion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four-channel 16-bit unsigned integer erosion, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single-channel 32-bit floating-point erosion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three-channel 32-bit floating-point erosion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four-channel 32-bit floating-point erosion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four-channel 32-bit floating-point erosion, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single-channel 8-bit unsigned integer erosion with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErodeBorder_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three-channel 8-bit unsigned integer erosion with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErodeBorder_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 8-bit unsigned integer erosion with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErodeBorder_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 8-bit unsigned integer erosion with border control, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErodeBorder_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single-channel 16-bit unsigned integer erosion with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErodeBorder_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three-channel 16-bit unsigned integer erosion with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErodeBorder_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 16-bit unsigned integer erosion with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErodeBorder_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 16-bit unsigned integer erosion with border control, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErodeBorder_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single-channel 32-bit floating-point erosion with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErodeBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three-channel 32-bit floating-point erosion with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErodeBorder_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 32-bit floating-point erosion with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErodeBorder_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 32-bit floating-point erosion with border control, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErodeBorder_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single-channel 8-bit unsigned integer 3x3 erosion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel 8-bit unsigned integer 3x3 erosion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 8-bit unsigned integer 3x3 erosion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 8-bit unsigned integer 3x3 erosion, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single-channel 16-bit unsigned integer 3x3 erosion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel 16-bit unsigned integer 3x3 erosion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 16-bit unsigned integer 3x3 erosion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 16-bit unsigned integer 3x3 erosion, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single-channel 32-bit floating-point 3x3 erosion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel 32-bit floating-point 3x3 erosion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 32-bit floating-point 3x3 erosion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 32-bit floating-point 3x3 erosion, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single-channel 64-bit floating-point 3x3 erosion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3_64f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single-channel 8-bit unsigned integer 3x3 erosion with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3Border_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three-channel 8-bit unsigned integer 3x3 erosion with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3Border_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 8-bit unsigned integer 3x3 erosion with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3Border_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 8-bit unsigned integer 3x3 erosion with border control, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3Border_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single-channel 16-bit unsigned integer 3x3 erosion with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3Border_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three-channel 16-bit unsigned integer 3x3 erosion with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3Border_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 16-bit unsigned integer 3x3 erosion with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3Border_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 16-bit unsigned integer 3x3 erosion with border control, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3Border_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single-channel 32-bit floating-point 3x3 erosion with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3Border_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three-channel 32-bit floating-point 3x3 erosion with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3Border_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 32-bit floating-point 3x3 erosion with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3Border_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four-channel 32-bit floating-point 3x3 erosion with border control, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">Source image starting point relative to pSrc.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(mDll, SetLastError = true)]
        public static extern NppStatus nppiErode3x3Border_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

    }
}