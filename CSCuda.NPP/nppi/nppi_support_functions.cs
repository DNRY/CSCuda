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
        /// 8-bit unsigned image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_8u_C1(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 2 channel 8-bit unsigned image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_8u_C2(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 3 channel 8-bit unsigned image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_8u_C3(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 4 channel 8-bit unsigned image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_8u_C4(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 16-bit unsigned image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_16u_C1(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 2 channel 16-bit unsigned image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_16u_C2(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 3 channel 16-bit unsigned image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_16u_C3(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 4 channel 16-bit unsigned image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_16u_C4(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 16-bit signed image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_16s_C1(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 2 channel 16-bit signed image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_16s_C2(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 4 channel 16-bit signed image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_16s_C4(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 1 channel 16-bit signed complex image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_16sc_C1(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 2 channel 16-bit signed complex image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_16sc_C2(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 3 channel 16-bit signed complex image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_16sc_C3(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 4 channel 16-bit signed complex image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_16sc_C4(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 32-bit signed image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_32s_C1(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 3 channel 32-bit signed image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_32s_C3(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 4 channel 32-bit signed image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_32s_C4(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 32-bit integer complex image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_32sc_C1(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 2 channel 32-bit integer complex image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_32sc_C2(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 3 channel 32-bit integer complex image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_32sc_C3(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 4 channel 32-bit integer complex image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_32sc_C4(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 32-bit floating point image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_32f_C1(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 2 channel 32-bit floating point image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_32f_C2(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 3 channel 32-bit floating point image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_32f_C3(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 4 channel 32-bit floating point image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_32f_C4(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 32-bit float complex image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_32fc_C1(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 2 channel 32-bit float complex image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_32fc_C2(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 3 channel 32-bit float complex image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_32fc_C3(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// 4 channel 32-bit float complex image memory allocator.
        /// </summary>
        /// <param name="nWidthPixels">Image width.</param>
        /// <param name="nHeightPixels">Image height.</param>
        /// <param name="pStepBytes">line_step.</param>
        /// <returns>Pointer to new image data.</returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern IntPtr nppiMalloc_32fc_C4(
            int nWidthPixels,
            int nHeightPixels,
            ref int pStepBytes);

        /// <summary>
        /// Free method for any 2D allocated memory. This method should be used to free memory allocated with any of the nppiMalloc_<modifier> methods.
        /// </summary>
        /// <param name="pData">A pointer to memory allocated using nppiMalloc_<modifier>.</param>
        /// <returns></returns>
        [DllImport(suDll, SetLastError = true)]
        public static extern void nppiFree(
            IntPtr pData);

    }
}