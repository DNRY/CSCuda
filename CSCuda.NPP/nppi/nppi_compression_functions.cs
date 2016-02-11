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
        /// Apply quality factor to raw 8-bit quantization table.
        /// This is effectively and in-place method that modifies a given raw quantization table based on a quality factor. Note that this method is a host method and that the pointer to the raw quantization table is a host pointer.
        /// </summary>
        /// <param name="hpQuantRawTable">Raw quantization table.</param>
        /// <param name="nQualityFactor">Quality factor for the table. Range is [1:100].</param>
        /// <returns>Error code: ::NPP_NULL_POINTER_ERROR is returned if hpQuantRawTable is 0.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiQuantFwdRawTableInit_JPEG_8u(
            IntPtr hpQuantRawTable,
            int nQualityFactor);

        /// <summary>
        /// Initializes a quantization table for
        /// This method is a host method. It consumes and produces host data. I.e. the pointers passed to this function must be host pointers. The resulting table needs to be transferred to device memory in order to be used with
        /// </summary>
        /// <param name="hpQuantRawTable">Host pointer to raw quantization table as returned by</param>
        /// <param name="hpQuantFwdRawTable">Forward quantization table for use with</param>
        /// <returns>Error code: ::NPP_NULL_POINTER_ERROR pQuantRawTable is 0.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiQuantFwdTableInit_JPEG_8u16u(
            IntPtr hpQuantRawTable,
            IntPtr hpQuantFwdRawTable);

        /// <summary>
        /// Initializes a quantization table for
        /// This method is a host method and consumes and produces host data. I.e. the pointers passed to this function must be host pointers. The resulting table needs to be transferred to device memory in order to be used with
        /// </summary>
        /// <param name="hpQuantRawTable">Raw quantization table.</param>
        /// <param name="hpQuantFwdRawTable">Inverse quantization table.</param>
        /// <returns>::NPP_NULL_POINTER_ERROR pQuantRawTable or pQuantFwdRawTable is0.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiQuantInvTableInit_JPEG_8u16u(
            IntPtr hpQuantRawTable,
            IntPtr hpQuantFwdRawTable);

        /// <summary>
        /// Forward DCT, quantization and level shift part of the JPEG encoding. Input is expected in 8x8 macro blocks and output is expected to be in 64x1 macro blocks.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pQuantFwdTable">Forward quantization tables for JPEG encoding created using</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>Error codes:</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            IntPtr pQuantFwdTable,
            NppiSize oSizeROI);

        /// <summary>
        /// Inverse DCT, de-quantization and level shift part of the JPEG decoding. Input is expected in 64x1 macro blocks and output is expected to be in 8x8 macro blocks.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">Image width in pixels x 8 x sizeof(Npp16s).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">Image width in pixels x 8 x sizeof(Npp16s).</param>
        /// <param name="pQuantInvTable">Inverse quantization tables for JPEG decoding created using</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>Error codes:</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            IntPtr pQuantInvTable,
            NppiSize oSizeROI);

        /// <summary>
        /// Initializes DCT state structure and allocates additional resources.
        /// </summary>
        /// <param name="ppState">Pointer to pointer to DCT state structure.</param>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDCTInitAlloc(
            IntPtr ppState);

        /// <summary>
        /// Frees the additional resources of the DCT state structure.
        /// </summary>
        /// <param name="pState">Pointer to DCT state structure.</param>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDCTFree(
            IntPtr pState);

        /// <summary>
        /// Forward DCT, quantization and level shift part of the JPEG encoding. Input is expected in 8x8 macro blocks and output is expected to be in 64x1 macro blocks. The new version of the primitive takes the ROI in image pixel size and works with DCT coefficients that are in zig-zag order.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">Image width in pixels x 8 x sizeof(Npp16s).</param>
        /// <param name="pQuantizationTable">Quantization Table in zig-zag order.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pState">Pointer to DCT state structure. This structure must be initialized allocated and initialized using</param>
        /// <returns>Error codes:</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            IntPtr pQuantizationTable,
            NppiSize oSizeROI,
            IntPtr pState);

        /// <summary>
        /// Inverse DCT, de-quantization and level shift part of the JPEG decoding. Input is expected in 64x1 macro blocks and output is expected to be in 8x8 macro blocks. The new version of the primitive takes the ROI in image pixel size and works with DCT coefficients that are in zig-zag order.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">Image width in pixels x 8 x sizeof(Npp16s).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pQuantizationTable">Quantization Table in zig-zag order.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pState">Pointer to DCT state structure. This structure must be initialized allocated and initialized using</param>
        /// <returns>Error codes:</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R_NEW(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            IntPtr pQuantizationTable,
            NppiSize oSizeROI,
            IntPtr pState);

        /// <summary>
        /// Returns the length of the NppiDecodeHuffmanSpec structure.
        /// </summary>
        /// <param name="pSize">Pointer to a variable that will receive the length of the NppiDecodeHuffmanSpec structure.</param>
        /// <returns>Error codes:</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDecodeHuffmanSpecGetBufSize_JPEG(
            IntPtr pSize);

        /// <summary>
        /// Creates a Huffman table in a format that is suitable for the decoder on the host.
        /// </summary>
        /// <param name="pRawHuffmanTable">Huffman table formated as specified in the JPEG standard.</param>
        /// <param name="eTableType">Enum specifying type of table (nppiDCTable or nppiACTable).</param>
        /// <param name="pHuffmanSpec">Pointer to the Huffman table for the decoder</param>
        /// <returns>Error codes:</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDecodeHuffmanSpecInitHost_JPEG(
            IntPtr pRawHuffmanTable,
            NppiHuffmanTableType eTableType,
            IntPtr pHuffmanSpec);

        /// <summary>
        /// Allocates memory and creates a Huffman table in a format that is suitable for the decoder on the host.
        /// </summary>
        /// <param name="pRawHuffmanTable">Huffman table formated as specified in the JPEG standard.</param>
        /// <param name="eTableType">Enum specifying type of table (nppiDCTable or nppiACTable).</param>
        /// <param name="ppHuffmanSpec">Pointer to returned pointer to the Huffman table for the decoder</param>
        /// <returns>Error codes:</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDecodeHuffmanSpecInitAllocHost_JPEG(
            IntPtr pRawHuffmanTable,
            NppiHuffmanTableType eTableType,
            IntPtr ppHuffmanSpec);

        /// <summary>
        /// Frees the host memory allocated by nppiDecodeHuffmanSpecInitAllocHost_JPEG.
        /// </summary>
        /// <param name="pHuffmanSpec">Pointer to the Huffman table for the decoder</param>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDecodeHuffmanSpecFreeHost_JPEG(
            IntPtr pHuffmanSpec);

        /// <summary>
        /// Huffman Decoding of the JPEG decoding on the host. Input is expected in byte stuffed huffman encoded JPEG scan and output is expected to be 64x1 macro blocks.
        /// </summary>
        /// <param name="pSrc">Byte-stuffed huffman encoded JPEG scan.</param>
        /// <param name="nLength">Byte length of the input.</param>
        /// <param name="restartInterval">Restart Interval, see JPEG standard.</param>
        /// <param name="Ss">Start Coefficient, see JPEG standard.</param>
        /// <param name="Se">End Coefficient, see JPEG standard.</param>
        /// <param name="Ah">Bit Approximation High, see JPEG standard.</param>
        /// <param name="Al">Bit Approximation Low, see JPEG standard.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pHuffmanTableDC">DC Huffman table.</param>
        /// <param name="pHuffmanTableAC">AC Huffman table.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>Error codes:</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDecodeHuffmanScanHost_JPEG_8u16s_P1R(
            IntPtr pSrc,
            Npp32s nLength,
            Npp32s restartInterval,
            Npp32s Ss,
            Npp32s Se,
            Npp32s Ah,
            Npp32s Al,
            IntPtr pDst,
            Npp32s nDstStep,
            IntPtr pHuffmanTableDC,
            IntPtr pHuffmanTableAC,
            NppiSize oSizeROI);

        /// <summary>
        /// Huffman Decoding of the JPEG decoding on the host. Input is expected in byte stuffed huffman encoded JPEG scan and output is expected to be 64x1 macro blocks.
        /// </summary>
        /// <param name="pSrc">Byte-stuffed huffman encoded JPEG scan.</param>
        /// <param name="nLength">Byte length of the input.</param>
        /// <param name="nRestartInterval">Restart Interval, see JPEG standard.</param>
        /// <param name="nSs">Start Coefficient, see JPEG standard.</param>
        /// <param name="nSe">End Coefficient, see JPEG standard.</param>
        /// <param name="nAh">Bit Approximation High, see JPEG standard.</param>
        /// <param name="nAl">Bit Approximation Low, see JPEG standard.</param>
        /// <param name="apDst">destination_image_pointer.</param>
        /// <param name="aDstStep">destination_image_line_step.</param>
        /// <param name="apHuffmanDCTable">DC Huffman tables.</param>
        /// <param name="apHuffmanACTable">AC Huffman tables.</param>
        /// <param name="aSizeROI">roi_specification.</param>
        /// <returns>Error codes:</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDecodeHuffmanScanHost_JPEG_8u16s_P3R(
            IntPtr pSrc,
            Npp32s nLength,
            Npp32s nRestartInterval,
            Npp32s nSs,
            Npp32s nSe,
            Npp32s nAh,
            Npp32s nAl,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] apDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aDstStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] apHuffmanDCTable,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] apHuffmanACTable,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]NppiSize[] aSizeROI);

        /// <summary>
        /// Returns the length of the NppiEncodeHuffmanSpec structure.
        /// </summary>
        /// <param name="pSize">Pointer to a variable that will receive the length of the NppiEncodeHuffmanSpec structure.</param>
        /// <returns>Error codes:</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiEncodeHuffmanSpecGetBufSize_JPEG(
            IntPtr pSize);

        /// <summary>
        /// Creates a Huffman table in a format that is suitable for the encoder.
        /// </summary>
        /// <param name="pRawHuffmanTable">Huffman table formated as specified in the JPEG standard.</param>
        /// <param name="eTableType">Enum specifying type of table (nppiDCTable or nppiACTable).</param>
        /// <param name="pHuffmanSpec">Pointer to the Huffman table for the decoder</param>
        /// <returns>Error codes:</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiEncodeHuffmanSpecInit_JPEG(
            IntPtr pRawHuffmanTable,
            NppiHuffmanTableType eTableType,
            IntPtr pHuffmanSpec);

        /// <summary>
        /// Allocates memory and creates a Huffman table in a format that is suitable for the encoder.
        /// </summary>
        /// <param name="pRawHuffmanTable">Huffman table formated as specified in the JPEG standard.</param>
        /// <param name="eTableType">Enum specifying type of table (nppiDCTable or nppiACTable).</param>
        /// <param name="ppHuffmanSpec">Pointer to returned pointer to the Huffman table for the encoder</param>
        /// <returns>Error codes:</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiEncodeHuffmanSpecInitAlloc_JPEG(
            IntPtr pRawHuffmanTable,
            NppiHuffmanTableType eTableType,
            IntPtr ppHuffmanSpec);

        /// <summary>
        /// Frees the memory allocated by nppiEncodeHuffmanSpecInitAlloc_JPEG.
        /// </summary>
        /// <param name="pHuffmanSpec">Pointer to the Huffman table for the encoder</param>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiEncodeHuffmanSpecFree_JPEG(
            IntPtr pHuffmanSpec);

        /// <summary>
        /// Huffman Encoding of the JPEG Encoding. Input is expected to be 64x1 macro blocks and output is expected as byte stuffed huffman encoded JPEG scan.
        /// </summary>
        /// <param name="pSrc">destination_image_pointer.</param>
        /// <param name="nSrcStep">destination_image_line_step.</param>
        /// <param name="nRestartInterval">Restart Interval, see JPEG standard. Currently only values <=0 are supported.</param>
        /// <param name="nSs">Start Coefficient, see JPEG standard.</param>
        /// <param name="nSe">End Coefficient, see JPEG standard.</param>
        /// <param name="nAh">Bit Approximation High, see JPEG standard.</param>
        /// <param name="nAl">Bit Approximation Low, see JPEG standard.</param>
        /// <param name="pDst">Byte-stuffed huffman encoded JPEG scan.</param>
        /// <param name="nLength">Byte length of the huffman encoded JPEG scan.</param>
        /// <param name="pHuffmanTableDC">DC Huffman table.</param>
        /// <param name="pHuffmanTableAC">AC Huffman table.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>Error codes:</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiEncodeHuffmanScan_JPEG_8u16s_P1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            Npp32s restartInterval,
            Npp32s Ss,
            Npp32s Se,
            Npp32s Ah,
            Npp32s Al,
            IntPtr pDst,
            IntPtr nLength,
            IntPtr pHuffmanTableDC,
            IntPtr pHuffmanTableAC,
            NppiSize oSizeROI,
            IntPtr pTempStorage);

        /// <summary>
        /// Huffman Encoding of the JPEG Encoding. Input is expected to be 64x1 macro blocks and output is expected as byte stuffed huffman encoded JPEG scan.
        /// </summary>
        /// <param name="apSrc">destination_image_pointer.</param>
        /// <param name="aSrcStep">destination_image_line_step.</param>
        /// <param name="nRestartInterval">Restart Interval, see JPEG standard. Currently only values <=0 are supported.</param>
        /// <param name="nSs">Start Coefficient, see JPEG standard.</param>
        /// <param name="nSe">End Coefficient, see JPEG standard.</param>
        /// <param name="nAh">Bit Approximation High, see JPEG standard.</param>
        /// <param name="nAl">Bit Approximation Low, see JPEG standard.</param>
        /// <param name="pDst">Byte-stuffed huffman encoded JPEG scan.</param>
        /// <param name="nLength">Byte length of the huffman encoded JPEG scan.</param>
        /// <param name="apHuffmanTableDC">DC Huffman tables.</param>
        /// <param name="apHuffmanTableAC">AC Huffman tables.</param>
        /// <param name="aSizeROI">roi_specification.</param>
        /// <returns>Error codes:</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiEncodeHuffmanScan_JPEG_8u16s_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] apSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aSrcStep,
            Npp32s nRestartInterval,
            Npp32s nSs,
            Npp32s nSe,
            Npp32s nAh,
            Npp32s nAl,
            IntPtr pDst,
            IntPtr nLength,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] apHuffmanDCTable,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] apHuffmanACTable,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]NppiSize[] aSizeROI,
            IntPtr pTempStorage);

        /// <summary>
        /// Optimize Huffman Encoding of the JPEG Encoding. Input is expected to be 64x1 macro blocks and output is expected as byte stuffed huffman encoded JPEG scan.
        /// </summary>
        /// <param name="pSrc">destination_image_pointer.</param>
        /// <param name="nSrcStep">destination_image_line_step.</param>
        /// <param name="nRestartInterval">Restart Interval, see JPEG standard. Currently only values <=0 are supported.</param>
        /// <param name="nSs">Start Coefficient, see JPEG standard.</param>
        /// <param name="nSe">End Coefficient, see JPEG standard.</param>
        /// <param name="nAh">Bit Approximation High, see JPEG standard.</param>
        /// <param name="nAl">Bit Approximation Low, see JPEG standard.</param>
        /// <param name="pDst">Byte-stuffed huffman encoded JPEG scan.</param>
        /// <param name="pLength">Pointer to the byte length of the huffman encoded JPEG scan.</param>
        /// <param name="hpCodesDC">Host pointer to the code of the huffman tree for DC component.</param>
        /// <param name="hpTableDC">Host pointer to the table of the huffman tree for DC component.</param>
        /// <param name="hpCodesAC">Host pointer to the code of the huffman tree for AC component.</param>
        /// <param name="hpTableAC">Host pointer to the table of the huffman tree for AC component.</param>
        /// <param name="pHuffmanTableDC">DC Huffman table.</param>
        /// <param name="pHuffmanTableAC">AC Huffman table.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>Error codes:</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiEncodeOptimizeHuffmanScan_JPEG_8u16s_P1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            Npp32s nRestartInterval,
            Npp32s nSs,
            Npp32s nSe,
            Npp32s nAh,
            Npp32s nAl,
            IntPtr pDst,
            IntPtr pLength,
            IntPtr hpCodesDC,
            IntPtr hpTableDC,
            IntPtr hpCodesAC,
            IntPtr hpTableAC,
            IntPtr aHuffmanDCTable,
            IntPtr aHuffmanACTable,
            NppiSize oSizeROI,
            IntPtr pTempStorage);

        /// <summary>
        /// Optimize Huffman Encoding of the JPEG Encoding. Input is expected to be 64x1 macro blocks and output is expected as byte stuffed huffman encoded JPEG scan.
        /// </summary>
        /// <param name="apSrc">destination_image_pointer.</param>
        /// <param name="aSrcStep">destination_image_line_step.</param>
        /// <param name="nRestartInterval">Restart Interval, see JPEG standard. Currently only values <=0 are supported.</param>
        /// <param name="nSs">Start Coefficient, see JPEG standard.</param>
        /// <param name="nSe">End Coefficient, see JPEG standard.</param>
        /// <param name="nAh">Bit Approximation High, see JPEG standard.</param>
        /// <param name="nAl">Bit Approximation Low, see JPEG standard.</param>
        /// <param name="pDst">Byte-stuffed huffman encoded JPEG scan.</param>
        /// <param name="pLength">Pointer to the byte length of the huffman encoded JPEG scan.</param>
        /// <param name="hpCodesDC">Host pointer to the code of the huffman tree for DC component.</param>
        /// <param name="hpTableDC">Host pointer to the table of the huffman tree for DC component.</param>
        /// <param name="hpCodesAC">Host pointer to the code of the huffman tree for AC component.</param>
        /// <param name="hpTableAC">Host pointer to the table of the huffman tree for AC component.</param>
        /// <param name="apHuffmanTableDC">DC Huffman tables.</param>
        /// <param name="apHuffmanTableAC">AC Huffman tables.</param>
        /// <param name="aSizeROI">roi_specification.</param>
        /// <returns>Error codes:</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiEncodeOptimizeHuffmanScan_JPEG_8u16s_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aSrcStep,
            Npp32s nRestartInterval,
            Npp32s nSs,
            Npp32s nSe,
            Npp32s nAh,
            Npp32s nAl,
            IntPtr pDst,
            IntPtr pLength,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] hpCodesDC,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] hpTableDC,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] hpCodesAC,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] hpTableAC,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] aHuffmanDCTable,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] aHuffmanACTable,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]NppiSize[] oSizeROI,
            IntPtr pTempStorage);

        /// <summary>
        /// Calculates the size of the temporary buffer for baseline Huffman encoding.
        /// </summary>
        /// <param name="oSize">Image Dimension.</param>
        /// <param name="pBufSize">Pointer to variable that returns the size of the temporary buffer.</param>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiEncodeHuffmanGetSize(
            NppiSize oSize,
            int nChannels,
            IntPtr pBufSize);

        /// <summary>
        /// Calculates the size of the temporary buffer for optimize Huffman coding.
        /// See nppiGenerateOptimizeHuffmanTable_JPEG.
        /// </summary>
        /// <param name="oSize">Image Dimension.</param>
        /// <param name="nChannels">Number of channels in the image.</param>
        /// <param name="pBufSize">Pointer to variable that returns the size of the temporary buffer.</param>
        /// <returns>NPP_SUCCESS Indicates no error. Any other value indicates an error or a warning</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiEncodeOptimizeHuffmanGetSize(
            NppiSize oSize,
            int nChannels,
            IntPtr pBufSize);

    }
}