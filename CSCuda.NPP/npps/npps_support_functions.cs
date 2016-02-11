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
        /// 8-bit unsigned signal allocator.
        /// </summary>
        /// <param name="nSize">Number of unsigned chars in the new signal.</param>
        /// <returns>A pointer to the new signal. 0 (NULL-pointer) indicates that an error occurred during allocation.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern IntPtr nppsMalloc_8u(
            int nSize);

        /// <summary>
        /// 8-bit signed signal allocator.
        /// </summary>
        /// <param name="nSize">Number of (signed) chars in the new signal.</param>
        /// <returns>A pointer to the new signal. 0 (NULL-pointer) indicates that an error occurred during allocation.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern IntPtr nppsMalloc_8s(
            int nSize);

        /// <summary>
        /// 16-bit unsigned signal allocator.
        /// </summary>
        /// <param name="nSize">Number of unsigned shorts in the new signal.</param>
        /// <returns>A pointer to the new signal. 0 (NULL-pointer) indicates that an error occurred during allocation.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern IntPtr nppsMalloc_16u(
            int nSize);

        /// <summary>
        /// 16-bit signal allocator.
        /// </summary>
        /// <param name="nSize">Number of shorts in the new signal.</param>
        /// <returns>A pointer to the new signal. 0 (NULL-pointer) indicates that an error occurred during allocation.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern IntPtr nppsMalloc_16s(
            int nSize);

        /// <summary>
        /// 16-bit complex-value signal allocator.
        /// </summary>
        /// <param name="nSize">Number of 16-bit complex numbers in the new signal.</param>
        /// <returns>A pointer to the new signal. 0 (NULL-pointer) indicates that an error occurred during allocation.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern IntPtr nppsMalloc_16sc(
            int nSize);

        /// <summary>
        /// 32-bit unsigned signal allocator.
        /// </summary>
        /// <param name="nSize">Number of unsigned ints in the new signal.</param>
        /// <returns>A pointer to the new signal. 0 (NULL-pointer) indicates that an error occurred during allocation.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern IntPtr nppsMalloc_32u(
            int nSize);

        /// <summary>
        /// 32-bit integer signal allocator.
        /// </summary>
        /// <param name="nSize">Number of ints in the new signal.</param>
        /// <returns>A pointer to the new signal. 0 (NULL-pointer) indicates that an error occurred during allocation.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern IntPtr nppsMalloc_32s(
            int nSize);

        /// <summary>
        /// 32-bit complex integer signal allocator.
        /// </summary>
        /// <param name="nSize">Number of complex integner values in the new signal.</param>
        /// <returns>A pointer to the new signal. 0 (NULL-pointer) indicates that an error occurred during allocation.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern IntPtr nppsMalloc_32sc(
            int nSize);

        /// <summary>
        /// 32-bit float signal allocator.
        /// </summary>
        /// <param name="nSize">Number of floats in the new signal.</param>
        /// <returns>A pointer to the new signal. 0 (NULL-pointer) indicates that an error occurred during allocation.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern IntPtr nppsMalloc_32f(
            int nSize);

        /// <summary>
        /// 32-bit complex float signal allocator.
        /// </summary>
        /// <param name="nSize">Number of complex float values in the new signal.</param>
        /// <returns>A pointer to the new signal. 0 (NULL-pointer) indicates that an error occurred during allocation.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern IntPtr nppsMalloc_32fc(
            int nSize);

        /// <summary>
        /// 64-bit long integer signal allocator.
        /// </summary>
        /// <param name="nSize">Number of long ints in the new signal.</param>
        /// <returns>A pointer to the new signal. 0 (NULL-pointer) indicates that an error occurred during allocation.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern IntPtr nppsMalloc_64s(
            int nSize);

        /// <summary>
        /// 64-bit complex long integer signal allocator.
        /// </summary>
        /// <param name="nSize">Number of complex long int values in the new signal.</param>
        /// <returns>A pointer to the new signal. 0 (NULL-pointer) indicates that an error occurred during allocation.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern IntPtr nppsMalloc_64sc(
            int nSize);

        /// <summary>
        /// 64-bit float (double) signal allocator.
        /// </summary>
        /// <param name="nSize">Number of doubles in the new signal.</param>
        /// <returns>A pointer to the new signal. 0 (NULL-pointer) indicates that an error occurred during allocation.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern IntPtr nppsMalloc_64f(
            int nSize);

        /// <summary>
        /// 64-bit complex complex signal allocator.
        /// </summary>
        /// <param name="nSize">Number of complex double valuess in the new signal.</param>
        /// <returns>A pointer to the new signal. 0 (NULL-pointer) indicates that an error occurred during allocation.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern IntPtr nppsMalloc_64fc(
            int nSize);

        /// <summary>
        /// Free method for any signal memory.
        /// </summary>
        /// <param name="pValues">A pointer to memory allocated using nppiMalloc_"modifier".</param>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern void nppsFree(
            IntPtr pValues);

    }
}