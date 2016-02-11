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
        /// </summary>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDstRect">User supplied host memory pointer to an</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGetResizeRect(
            NppiRect oSrcROI,
            IntPtr pDstRect,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 1 channel 8-bit unsigned image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_8u_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 3 channel 8-bit unsigned image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_8u_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 4 channel 8-bit unsigned image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_8u_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 4 channel 8-bit unsigned image resize not affecting alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_8u_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 3 channel 8-bit unsigned planar image resize.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 4 channel 8-bit unsigned planar image resize.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_8u_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 1 channel 16-bit unsigned image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_16u_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 3 channel 16-bit unsigned image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_16u_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 4 channel 16-bit unsigned image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_16u_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 4 channel 16-bit unsigned image resize not affecting alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_16u_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 3 channel 16-bit unsigned planar image resize.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_16u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 4 channel 16-bit unsigned planar image resize.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_16u_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 1 channel 16-bit signed image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_16s_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 3 channel 16-bit signed image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_16s_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 4 channel 16-bit signed image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_16s_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 4 channel 16-bit signed image resize not affecting alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_16s_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 3 channel 16-bit signed planar image resize.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_16s_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 4 channel 16-bit signed planar image resize.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_16s_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 1 channel 32-bit floating point image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_32f_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 3 channel 32-bit floating point image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_32f_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 4 channel 32-bit floating point image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_32f_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 4 channel 32-bit floating point image resize not affecting alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_32f_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 3 channel 32-bit floating point planar image resize.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_32f_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 4 channel 32-bit floating point planar image resize.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_32f_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 1 channel 64-bit floating point image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_64f_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 3 channel 64-bit floating point image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_64f_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 4 channel 64-bit floating point image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_64f_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 4 channel 64-bit floating point image resize not affecting alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_64f_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 3 channel 64-bit floating point planar image resize.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_64f_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// 4 channel 64-bit floating point planar image resize.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_64f_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            double nXShift,
            double nYShift,
            int eInterpolation);

        /// <summary>
        /// Buffer size for
        /// </summary>
        /// <param name="oSrcROI">roi_specification.</param>
        /// <param name="oDstROI">roi_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <param name="eInterpolationMode">The type of eInterpolation to perform resampling. Currently only supports NPPI_INTER_LANCZOS3_Advanced.</param>
        /// <returns>NPP_NULL_POINTER_ERROR if hpBufferSize is 0 (NULL), roi_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeAdvancedGetBufferHostSize_8u_C1R(
            NppiSize oSrcROI,
            NppiSize oDstROI,
            IntPtr hpBufferSize,
            int eInterpolationMode);

        /// <summary>
        /// 1 channel 8-bit unsigned image resize. This primitive matches the behavior of GraphicsMagick++.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="pBuffer">Device buffer that is used during calculations.</param>
        /// <param name="eInterpolationMode">The type of eInterpolation to perform resampling. Currently only supports NPPI_INTER_LANCZOS3_Advanced.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResizeSqrPixel_8u_C1R_Advanced(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nXFactor,
            double nYFactor,
            IntPtr pBuffer,
            int eInterpolationMode);

        /// <summary>
        /// 1 channel 8-bit unsigned image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="dstROISize">Size in pixels of the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResize_8u_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize dstROISize,
            double nXFactor,
            double nYFactor,
            int eInterpolation);

        /// <summary>
        /// 3 channel 8-bit unsigned image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="dstROISize">Size in pixels of the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResize_8u_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize dstROISize,
            double nXFactor,
            double nYFactor,
            int eInterpolation);

        /// <summary>
        /// 4 channel 8-bit unsigned image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="dstROISize">Size in pixels of the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResize_8u_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize dstROISize,
            double nXFactor,
            double nYFactor,
            int eInterpolation);

        /// <summary>
        /// 4 channel 8-bit unsigned image resize not affecting alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="dstROISize">Size in pixels of the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResize_8u_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize dstROISize,
            double nXFactor,
            double nYFactor,
            int eInterpolation);

        /// <summary>
        /// 3 channel 8-bit unsigned planar image resize.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="dstROISize">Size in pixels of the destination image</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResize_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize dstROISize,
            double nXFactor,
            double nYFactor,
            int eInterpolation);

        /// <summary>
        /// 4 channel 8-bit unsigned planar image resize.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="dstROISize">Size in pixels of the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResize_8u_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiSize dstROISize,
            double nXFactor,
            double nYFactor,
            int eInterpolation);

        /// <summary>
        /// 1 channel 16-bit unsigned image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="dstROISize">Size in pixels of the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResize_16u_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize dstROISize,
            double nXFactor,
            double nYFactor,
            int eInterpolation);

        /// <summary>
        /// 3 channel 16-bit unsigned image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="dstROISize">Size in pixels of the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResize_16u_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize dstROISize,
            double nXFactor,
            double nYFactor,
            int eInterpolation);

        /// <summary>
        /// 4 channel 16-bit unsigned image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="dstROISize">Size in pixels of the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResize_16u_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize dstROISize,
            double nXFactor,
            double nYFactor,
            int eInterpolation);

        /// <summary>
        /// 4 channel 16-bit unsigned image resize not affecting alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="dstROISize">Size in pixels of the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResize_16u_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize dstROISize,
            double nXFactor,
            double nYFactor,
            int eInterpolation);

        /// <summary>
        /// 3 channel 16-bit unsigned planar image resize.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="dstROISize">Size in pixels of the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResize_16u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize dstROISize,
            double nXFactor,
            double nYFactor,
            int eInterpolation);

        /// <summary>
        /// 4 channel 16-bit unsigned planar image resize.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="dstROISize">Size in pixels of the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResize_16u_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiSize dstROISize,
            double nXFactor,
            double nYFactor,
            int eInterpolation);

        /// <summary>
        /// 1 channel 32-bit floating point image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="dstROISize">Size in pixels of the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResize_32f_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize dstROISize,
            double nXFactor,
            double nYFactor,
            int eInterpolation);

        /// <summary>
        /// 3 channel 32-bit floating point image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="dstROISize">Size in pixels of the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResize_32f_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize dstROISize,
            double nXFactor,
            double nYFactor,
            int eInterpolation);

        /// <summary>
        /// 4 channel 32-bit floating point image resize.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="dstROISize">Size in pixels of the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResize_32f_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize dstROISize,
            double nXFactor,
            double nYFactor,
            int eInterpolation);

        /// <summary>
        /// 4 channel 32-bit floating point image resize not affecting alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="dstROISize">Size in pixels of the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResize_32f_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize dstROISize,
            double nXFactor,
            double nYFactor,
            int eInterpolation);

        /// <summary>
        /// 3 channel 32-bit floating point planar image resize.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="dstROISize">Size in pixels of the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResize_32f_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize dstROISize,
            double nXFactor,
            double nYFactor,
            int eInterpolation);

        /// <summary>
        /// 4 channel 32-bit floating point planar image resize.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="dstROISize">Size in pixels of the destination image.</param>
        /// <param name="nXFactor">Factor by which x dimension is changed.</param>
        /// <param name="nYFactor">Factor by which y dimension is changed.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiResize_32f_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiSize dstROISize,
            double nXFactor,
            double nYFactor,
            int eInterpolation);

        /// <summary>
        /// 1 channel 8-bit unsigned image remap.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_8u_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 3 channel 8-bit unsigned image remap.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_8u_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 4 channel 8-bit unsigned image remap.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_8u_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 4 channel 8-bit unsigned image remap not affecting alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_8u_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 3 channel 8-bit unsigned planar image remap.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 4 channel 8-bit unsigned planar image remap.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_8u_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 1 channel 16-bit unsigned image remap.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_16u_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 3 channel 16-bit unsigned image remap.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_16u_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 4 channel 16-bit unsigned image remap.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_16u_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 4 channel 16-bit unsigned image remap not affecting alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_16u_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 3 channel 16-bit unsigned planar image remap.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_16u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 4 channel 16-bit unsigned planar image remap.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_16u_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 1 channel 16-bit signed image remap.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_16s_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 3 channel 16-bit signed image remap.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_16s_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 4 channel 16-bit signed image remap.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_16s_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 4 channel 16-bit signed image remap not affecting alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_16s_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 3 channel 16-bit signed planar image remap.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_16s_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 4 channel 16-bit signed planar image remap.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_16s_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 1 channel 32-bit floating point image remap.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_32f_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 3 channel 32-bit floating point image remap.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_32f_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 4 channel 32-bit floating point image remap.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_32f_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 4 channel 32-bit floating point image remap not affecting alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_32f_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 3 channel 32-bit floating point planar image remap.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_32f_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 4 channel 32-bit floating point planar image remap.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_32f_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 1 channel 64-bit floating point image remap.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_64f_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 3 channel 64-bit floating point image remap.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_64f_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 4 channel 64-bit floating point image remap.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_64f_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 4 channel 64-bit floating point image remap not affecting alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_64f_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 3 channel 64-bit floating point planar image remap.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_64f_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// 4 channel 64-bit floating point planar image remap.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image.</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image.</param>
        /// <param name="nXMapStep">pXMap image array line step in bytes.</param>
        /// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image.</param>
        /// <param name="nYMapStep">pYMap image array line step in bytes.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Region of interest size in the destination image.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRemap_64f_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pXMap,
            int nXMapStep,
            IntPtr pYMap,
            int nYMapStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int eInterpolation);

        /// <summary>
        /// Compute shape of rotated image.
        /// </summary>
        /// <param name="oSrcROI">Region-of-interest of the source image.</param>
        /// <param name="aQuad">Array of 2D points. These points are the locations of the corners of the rotated ROI.</param>
        /// <param name="nAngle">The rotation nAngle.</param>
        /// <param name="nShiftX">Post-rotation shift in x-direction</param>
        /// <param name="nShiftY">Post-rotation shift in y-direction</param>
        /// <returns>roi_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGetRotateQuad(
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aQuad,
            double nAngle,
            double nShiftX,
            double nShiftY);

        /// <summary>
        /// Compute bounding-box of rotated image.
        /// </summary>
        /// <param name="oSrcROI">Region-of-interest of the source image.</param>
        /// <param name="aBoundingBox">Two 2D points representing the bounding-box of the rotated image. All four points from nppiGetRotateQuad are contained inside the axis-aligned rectangle spanned by the the two points of this bounding box.</param>
        /// <param name="nAngle">The rotation angle.</param>
        /// <param name="nShiftX">Post-rotation shift in x-direction.</param>
        /// <param name="nShiftY">Post-rotation shift in y-direction.</param>
        /// <returns>roi_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGetRotateBound(
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aBoundingBox,
            double nAngle,
            double nShiftX,
            double nShiftY);

        /// <summary>
        /// Computes affine transform coefficients based on source ROI and destination quadrilateral.
        /// The function computes the coefficients of an affine transformation that maps the given source ROI (axis aligned rectangle with integer coordinates) to a quadrilateral in the destination image.
        /// An affine transform in 2D is fully determined by the mapping of just three vertices. This function's API allows for passing a complete quadrilateral effectively making the prolem overdetermined. What this means in practice is, that for certain quadrilaterals it is not possible to find an affine transform that would map all four corners of the source ROI to the four vertices of that quadrilateral.
        /// The function circumvents this problem by only looking at the first three vertices of the destination image quadrilateral to determine the affine transformation's coefficients. If the destination quadrilateral is indeed one that cannot be mapped using an affine transformation the functions informs the user of this situation by returning a
        /// </summary>
        /// <param name="oSrcROI">The source ROI. This rectangle needs to be at least one pixel wide and high. If either width or hight are less than one an ::NPP_RECT_ERROR is returned.</param>
        /// <param name="aQuad">The destination quadrilateral.</param>
        /// <param name="aCoeffs">The resulting affine transform coefficients.</param>
        /// <returns>Error codes:</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGetAffineTransform(
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aQuad,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs);

        /// <summary>
        /// Compute shape of transformed image.
        /// This method computes the quadrilateral in the destination image that the source ROI is transformed into by the affine transformation expressed by the coefficients array (aCoeffs).
        /// </summary>
        /// <param name="oSrcROI">The source ROI.</param>
        /// <param name="aQuad">The resulting destination quadrangle.</param>
        /// <param name="aCoeffs">The afine transform coefficients.</param>
        /// <returns>Error codes:</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGetAffineQuad(
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aQuad,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs);

        /// <summary>
        /// Compute bounding-box of transformed image.
        /// The method effectively computes the bounding box (axis aligned rectangle) of the transformed source ROI (see
        /// </summary>
        /// <param name="oSrcROI">The source ROI.</param>
        /// <param name="aBound">The resulting bounding box.</param>
        /// <param name="aCoeffs">The afine transform coefficients.</param>
        /// <returns>Error codes:</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGetAffineBound(
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aBound,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs);

        /// <summary>
        /// Calculates perspective transform coefficients given source rectangular ROI and its destination quadrangle projection
        /// </summary>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="quad">Destination quadrangle</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <returns>Error codes:</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGetPerspectiveTransform(
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] quad,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs);

        /// <summary>
        /// Calculates perspective transform projection of given source rectangular ROI
        /// </summary>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="quad">Destination quadrangle</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <returns>Error codes:</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGetPerspectiveQuad(
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] quad,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs);

        /// <summary>
        /// Calculates bounding box of the perspective transform projection of the given source rectangular ROI
        /// </summary>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="bound">Bounding box of the transformed source ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <returns>Error codes:</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGetPerspectiveBound(
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] bound,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs);

        /// <summary>
        /// 8-bit unsigned image rotate.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nAngle">The angle of rotation in degrees.</param>
        /// <param name="nShiftX">Shift along horizontal axis</param>
        /// <param name="nShiftY">Shift along vertical axis</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRotate_8u_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nAngle,
            double nShiftX,
            double nShiftY,
            int eInterpolation);

        /// <summary>
        /// 3 channel 8-bit unsigned image rotate.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nAngle">The angle of rotation in degrees.</param>
        /// <param name="nShiftX">Shift along horizontal axis</param>
        /// <param name="nShiftY">Shift along vertical axis</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRotate_8u_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nAngle,
            double nShiftX,
            double nShiftY,
            int eInterpolation);

        /// <summary>
        /// 4 channel 8-bit unsigned image rotate.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nAngle">The angle of rotation in degrees.</param>
        /// <param name="nShiftX">Shift along horizontal axis</param>
        /// <param name="nShiftY">Shift along vertical axis</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRotate_8u_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nAngle,
            double nShiftX,
            double nShiftY,
            int eInterpolation);

        /// <summary>
        /// 4 channel 8-bit unsigned image rotate ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nAngle">The angle of rotation in degrees.</param>
        /// <param name="nShiftX">Shift along horizontal axis</param>
        /// <param name="nShiftY">Shift along vertical axis</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRotate_8u_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nAngle,
            double nShiftX,
            double nShiftY,
            int eInterpolation);

        /// <summary>
        /// 16-bit unsigned image rotate.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nAngle">The angle of rotation in degrees.</param>
        /// <param name="nShiftX">Shift along horizontal axis</param>
        /// <param name="nShiftY">Shift along vertical axis</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRotate_16u_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nAngle,
            double nShiftX,
            double nShiftY,
            int eInterpolation);

        /// <summary>
        /// 3 channel 16-bit unsigned image rotate.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nAngle">The angle of rotation in degrees.</param>
        /// <param name="nShiftX">Shift along horizontal axis</param>
        /// <param name="nShiftY">Shift along vertical axis</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRotate_16u_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nAngle,
            double nShiftX,
            double nShiftY,
            int eInterpolation);

        /// <summary>
        /// 4 channel 16-bit unsigned image rotate.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nAngle">The angle of rotation in degrees.</param>
        /// <param name="nShiftX">Shift along horizontal axis</param>
        /// <param name="nShiftY">Shift along vertical axis</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRotate_16u_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nAngle,
            double nShiftX,
            double nShiftY,
            int eInterpolation);

        /// <summary>
        /// 4 channel 16-bit unsigned image rotate ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nAngle">The angle of rotation in degrees.</param>
        /// <param name="nShiftX">Shift along horizontal axis</param>
        /// <param name="nShiftY">Shift along vertical axis</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRotate_16u_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nAngle,
            double nShiftX,
            double nShiftY,
            int eInterpolation);

        /// <summary>
        /// 32-bit float image rotate.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nAngle">The angle of rotation in degrees.</param>
        /// <param name="nShiftX">Shift along horizontal axis</param>
        /// <param name="nShiftY">Shift along vertical axis</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRotate_32f_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nAngle,
            double nShiftX,
            double nShiftY,
            int eInterpolation);

        /// <summary>
        /// 3 channel 32-bit float image rotate.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nAngle">The angle of rotation in degrees.</param>
        /// <param name="nShiftX">Shift along horizontal axis</param>
        /// <param name="nShiftY">Shift along vertical axis</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRotate_32f_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nAngle,
            double nShiftX,
            double nShiftY,
            int eInterpolation);

        /// <summary>
        /// 4 channel 32-bit float image rotate.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nAngle">The angle of rotation in degrees.</param>
        /// <param name="nShiftX">Shift along horizontal axis</param>
        /// <param name="nShiftY">Shift along vertical axis</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRotate_32f_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nAngle,
            double nShiftX,
            double nShiftY,
            int eInterpolation);

        /// <summary>
        /// 4 channel 32-bit float image rotate ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Size in pixels of the source image</param>
        /// <param name="oSrcROI">Region of interest in the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Region of interest in the destination image.</param>
        /// <param name="nAngle">The angle of rotation in degrees.</param>
        /// <param name="nShiftX">Shift along horizontal axis</param>
        /// <param name="nShiftY">Shift along vertical axis</param>
        /// <param name="eInterpolation">The type of interpolation to perform resampling</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRotate_32f_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            double nAngle,
            double nShiftX,
            double nShiftY,
            int eInterpolation);

        /// <summary>
        /// 1 channel 8-bit unsigned image mirror.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 1 channel 8-bit unsigned in place image mirror.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_8u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 3 channel 8-bit unsigned image mirror.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 3 channel 8-bit unsigned in place image mirror.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_8u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 4 channel 8-bit unsigned image mirror.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">Distance in bytes between starts of consecutive lines of the destination image.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 4 channel 8-bit unsigned in place image mirror.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_8u_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 4 channel 8-bit unsigned image mirror not affecting alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">Distance in bytes between starts of consecutive lines of the destination image.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 4 channel 8-bit unsigned in place image mirror not affecting alpha.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_8u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 1 channel 16-bit unsigned image mirror.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 1 channel 16-bit unsigned in place image mirror.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_16u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 3 channel 16-bit unsigned image mirror.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 3 channel 16-bit unsigned in place image mirror.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_16u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 4 channel 16-bit unsigned image mirror.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">Distance in bytes between starts of consecutive lines of the destination image.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 4 channel 16-bit unsigned in place image mirror.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_16u_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 4 channel 16-bit unsigned image mirror not affecting alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">Distance in bytes between starts of consecutive lines of the destination image.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 4 channel 16-bit unsigned in place image mirror not affecting alpha.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_16u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 1 channel 16-bit signed image mirror.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 1 channel 16-bit signed in place image mirror.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_16s_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 3 channel 16-bit signed image mirror.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 3 channel 16-bit signed in place image mirror.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_16s_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 4 channel 16-bit signed image mirror.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">Distance in bytes between starts of consecutive lines of the destination image.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 4 channel 16-bit signed in place image mirror.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_16s_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 4 channel 16-bit signed image mirror not affecting alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">Distance in bytes between starts of consecutive lines of the destination image.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 4 channel 16-bit signed in place image mirror not affecting alpha.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_16s_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 1 channel 32-bit image mirror.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_32s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 1 channel 32-bit signed in place image mirror.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_32s_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 3 channel 32-bit image mirror.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_32s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 3 channel 32-bit signed in place image mirror.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_32s_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 4 channel 32-bit image mirror.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">Distance in bytes between starts of consecutive lines of the destination image.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_32s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 4 channel 32-bit signed in place image mirror.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_32s_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 4 channel 32-bit image mirror not affecting alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">Distance in bytes between starts of consecutive lines of the destination image.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_32s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 4 channel 32-bit signed in place image mirror not affecting alpha.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_32s_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 1 channel 32-bit float image mirror.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 1 channel 32-bit float in place image mirror.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_32f_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 3 channel 32-bit float image mirror.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 3 channel 32-bit float in place image mirror.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_32f_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 4 channel 32-bit float image mirror.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">Distance in bytes between starts of consecutive lines of the destination image.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 4 channel 32-bit float in place image mirror.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_32f_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 4 channel 32-bit float image mirror not affecting alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">Distance in bytes between starts of consecutive lines of the destination image.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// 4 channel 32-bit float in place image mirror not affecting alpha.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMirror_32f_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oROI,
            NppiAxis flip);

        /// <summary>
        /// Single-channel 8-bit unsigned affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_8u_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel 8-bit unsigned affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_8u_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 8-bit unsigned affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_8u_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 8-bit unsigned affine warp, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_8u_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 8-bit unsigned affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 8-bit unsigned affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_8u_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Single-channel 16-bit unsigned affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_16u_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel 16-bit unsigned affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_16u_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 16-bit unsigned affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_16u_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 16-bit unsigned affine warp, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_16u_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 16-bit unsigned affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_16u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 16-bit unsigned affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_16u_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Single-channel 32-bit signed affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_32s_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel 32-bit signed affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_32s_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit signed affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_32s_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit signed affine warp, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_32s_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 32-bit signed affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_32s_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 32-bit signed affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_32s_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Single-channel 32-bit floating-point affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_32f_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel 32-bit floating-point affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_32f_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit floating-point affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_32f_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit floating-point affine warp, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_32f_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 32-bit floating-point affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_32f_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 32-bit floating-point affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_32f_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Single-channel 64-bit floating-point affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_64f_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel 64-bit floating-point affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_64f_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 64-bit floating-point affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_64f_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 64-bit floating-point affine warp, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_64f_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 64-bit floating-point affine warp.
        /// </summary>
        /// <param name="aSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_64f_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] aSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] aDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 64-bit floating-point affine warp.
        /// </summary>
        /// <param name="aSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffine_64f_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] aSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] aDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Single-channel 8-bit unsigned integer backwards affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_8u_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel 8-bit unsigned integer backwards affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_8u_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 8-bit unsigned integer backwards affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_8u_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 8-bit unsigned integer backwards affine warp, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_8u_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 8-bit unsigned integer backwards affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 8-bit unsigned integer backwards affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_8u_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Single-channel 16-bit unsigned integer backwards affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_16u_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel 16-bit unsigned integer backwards affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_16u_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 16-bit unsigned integer backwards affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_16u_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 16-bit unsigned integer backwards affine warp, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_16u_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 16-bit unsigned integer backwards affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_16u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 16-bit unsigned integer backwards affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_16u_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Single-channel 32-bit signed integer backwards affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_32s_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel 32-bit signed integer backwards affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_32s_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit signed integer backwards affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_32s_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit signed integer backwards affine warp, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_32s_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 32-bit signed integer backwards affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_32s_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 32-bit signed integer backwards affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_32s_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Single-channel 32-bit floating-point backwards affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_32f_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel 32-bit floating-point backwards affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_32f_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit floating-point backwards affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_32f_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit floating-point backwards affine warp, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_32f_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 32-bit floating-point backwards affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_32f_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 32-bit floating-point backwards affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Affine transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineBack_32f_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Single-channel 32-bit floating-point quad-based affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_8u_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Three-channel 8-bit unsigned integer quad-based affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_8u_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel 8-bit unsigned integer quad-based affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_8u_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel 8-bit unsigned integer quad-based affine warp, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_8u_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 8-bit unsigned integer quad-based affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 8-bit unsigned integer quad-based affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_8u_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Single-channel 16-bit unsigned integer quad-based affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_16u_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Three-channel 16-bit unsigned integer quad-based affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_16u_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel 16-bit unsigned integer quad-based affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_16u_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel 16-bit unsigned integer quad-based affine warp, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_16u_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 16-bit unsigned integer quad-based affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_16u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 16-bit unsigned integer quad-based affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_16u_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Single-channel 32-bit signed integer quad-based affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_32s_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Three-channel 32-bit signed integer quad-based affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_32s_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit signed integer quad-based affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_32s_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit signed integer quad-based affine warp, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_32s_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 32-bit signed integer quad-based affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_32s_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 32-bit signed integer quad-based affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_32s_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Single-channel 32-bit floating-point quad-based affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_32f_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Three-channel 32-bit floating-point quad-based affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_32f_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit floating-point quad-based affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_32f_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit floating-point quad-based affine warp, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_32f_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 32-bit floating-point quad-based affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_32f_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 32-bit floating-point quad-based affine warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpAffineQuad_32f_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Single-channel 8-bit unsigned integer perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_8u_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel 8-bit unsigned integer perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_8u_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 8-bit unsigned integer perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_8u_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 8-bit unsigned integer perspective warp, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_8u_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 8-bit unsigned integer perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 8-bit unsigned integer perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_8u_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Single-channel 16-bit unsigned integer perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_16u_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel 16-bit unsigned integer perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_16u_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 16-bit unsigned integer perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_16u_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 16-bit unsigned integer perspective warp, igoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_16u_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 16-bit unsigned integer perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_16u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 16-bit unsigned integer perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_16u_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Single-channel 32-bit signed integer perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_32s_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel 32-bit signed integer perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_32s_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit signed integer perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_32s_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit signed integer perspective warp, igoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_32s_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 32-bit signed integer perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_32s_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 32-bit signed integer perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_32s_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Single-channel 32-bit floating-point perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_32f_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel 32-bit floating-point perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_32f_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit floating-point perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_32f_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit floating-point perspective warp, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_32f_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 32-bit floating-point perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_32f_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 32-bit floating-point perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspective_32f_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Single-channel 8-bit unsigned integer backwards perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_8u_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel 8-bit unsigned integer backwards perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_8u_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 8-bit unsigned integer backwards perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_8u_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 8-bit unsigned integer backwards perspective warp, igoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_8u_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 8-bit unsigned integer backwards perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 8-bit unsigned integer backwards perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_8u_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Single-channel 16-bit unsigned integer backwards perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_16u_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel 16-bit unsigned integer backwards perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_16u_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 16-bit unsigned integer backwards perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_16u_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 16-bit unsigned integer backwards perspective warp, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_16u_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 16-bit unsigned integer backwards perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_16u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 16-bit unsigned integer backwards perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_16u_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Single-channel 32-bit signed integer backwards perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_32s_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel 32-bit signed integer backwards perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_32s_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit signed integer backwards perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_32s_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit signed integer backwards perspective warp, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_32s_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 32-bit signed integer backwards perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_32s_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 32-bit signed integer backwards perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_32s_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Single-channel 32-bit floating-point backwards perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_32f_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel 32-bit floating-point backwards perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_32f_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit floating-point backwards perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_32f_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit floating-point backwards perspective warp, ignorning alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_32f_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 32-bit floating-point backwards perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_32f_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 32-bit floating-point backwards perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aCoeffs">Perspective transform coefficients</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveBack_32f_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]double[] aCoeffs,
            int eInterpolation);

        /// <summary>
        /// Single-channel 8-bit unsigned integer quad-based perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_8u_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Three-channel 8-bit unsigned integer quad-based perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_8u_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel 8-bit unsigned integer quad-based perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_8u_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel 8-bit unsigned integer quad-based perspective warp, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_8u_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 8-bit unsigned integer quad-based perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 8-bit unsigned integer quad-based perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_8u_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Single-channel 16-bit unsigned integer quad-based perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_16u_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Three-channel 16-bit unsigned integer quad-based perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_16u_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel 16-bit unsigned integer quad-based perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_16u_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel 16-bit unsigned integer quad-based perspective warp, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_16u_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 16-bit unsigned integer quad-based perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_16u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 16-bit unsigned integer quad-based perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_16u_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Single-channel 32-bit signed integer quad-based perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_32s_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Three-channel 32-bit signed integer quad-based perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_32s_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit signed integer quad-based perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_32s_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit signed integer quad-based perspective warp, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_32s_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 32-bit signed integer quad-based perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_32s_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 32-bit signed integer quad-based perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_32s_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Single-channel 32-bit floating-point quad-based perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_32f_C1R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Three-channel 32-bit floating-point quad-based perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_32f_C3R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit floating-point quad-based perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_32f_C4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel 32-bit floating-point quad-based perspective warp, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_32f_AC4R(
            IntPtr pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            IntPtr pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Three-channel planar 32-bit floating-point quad-based perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_32f_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

        /// <summary>
        /// Four-channel planar 32-bit floating-point quad-based perspective warp.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="oSrcSize">Size of source image in pixels</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcROI">Source ROI</param>
        /// <param name="aSrcQuad">Source quad.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstROI">Destination ROI</param>
        /// <param name="aDstQuad">Destination quad.</param>
        /// <param name="eInterpolation">Interpolation mode: can be NPPI_INTER_NN, NPPI_INTER_LINEAR or NPPI_INTER_CUBIC</param>
        /// <returns>image_data_error_codes, roi_error_codes,</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiWarpPerspectiveQuad_32f_P4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            NppiSize oSrcSize,
            int nSrcStep,
            NppiRect oSrcROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aSrcQuad,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiRect oDstROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]double[] aDstQuad,
            int eInterpolation);

    }
}