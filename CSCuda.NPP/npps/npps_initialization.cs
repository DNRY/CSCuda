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
        /// 8-bit unsigned char, vector set method.
        /// </summary>
        /// <param name="nValue">Value used to initialize the vector pDst.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSet_8u(
            Npp8u nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 8-bit signed char, vector set method.
        /// </summary>
        /// <param name="nValue">Value used to initialize the vector pDst.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSet_8s(
            Npp8s nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned integer, vector set method.
        /// </summary>
        /// <param name="nValue">Value used to initialize the vector pDst.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSet_16u(
            Npp16u nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit signed integer, vector set method.
        /// </summary>
        /// <param name="nValue">Value used to initialize the vector pDst.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSet_16s(
            Npp16s nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit integer complex, vector set method.
        /// </summary>
        /// <param name="nValue">Value used to initialize the vector pDst.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSet_16sc(
            Npp16sc nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit unsigned integer, vector set method.
        /// </summary>
        /// <param name="nValue">Value used to initialize the vector pDst.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSet_32u(
            Npp32u nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit signed integer, vector set method.
        /// </summary>
        /// <param name="nValue">Value used to initialize the vector pDst.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSet_32s(
            Npp32s nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit integer complex, vector set method.
        /// </summary>
        /// <param name="nValue">Value used to initialize the vector pDst.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSet_32sc(
            Npp32sc nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit float, vector set method.
        /// </summary>
        /// <param name="nValue">Value used to initialize the vector pDst.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSet_32f(
            Npp32f nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit float complex, vector set method.
        /// </summary>
        /// <param name="nValue">Value used to initialize the vector pDst.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSet_32fc(
            Npp32fc nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit long long integer, vector set method.
        /// </summary>
        /// <param name="nValue">Value used to initialize the vector pDst.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSet_64s(
            Npp64s nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit long long integer complex, vector set method.
        /// </summary>
        /// <param name="nValue">Value used to initialize the vector pDst.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSet_64sc(
            Npp64sc nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit double, vector set method.
        /// </summary>
        /// <param name="nValue">Value used to initialize the vector pDst.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSet_64f(
            Npp64f nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit double complex, vector set method.
        /// </summary>
        /// <param name="nValue">Value used to initialize the vector pDst.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSet_64fc(
            Npp64fc nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char, vector zero method.
        /// </summary>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsZero_8u(
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit integer, vector zero method.
        /// </summary>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsZero_16s(
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit integer complex, vector zero method.
        /// </summary>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsZero_16sc(
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit integer, vector zero method.
        /// </summary>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsZero_32s(
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit integer complex, vector zero method.
        /// </summary>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsZero_32sc(
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit float, vector zero method.
        /// </summary>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsZero_32f(
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit float complex, vector zero method.
        /// </summary>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsZero_32fc(
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit long long integer, vector zero method.
        /// </summary>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsZero_64s(
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit long long integer complex, vector zero method.
        /// </summary>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsZero_64sc(
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit double, vector zero method.
        /// </summary>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsZero_64f(
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit double complex, vector zero method.
        /// </summary>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsZero_64fc(
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char, vector copy method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsCopy_8u(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit signed short, vector copy method.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsCopy_16s(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit signed integer, vector copy method.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsCopy_32s(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit float, vector copy method.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsCopy_32f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit signed integer, vector copy method.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsCopy_64s(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit complex short, vector copy method.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsCopy_16sc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit complex signed integer, vector copy method.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsCopy_32sc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit complex float, vector copy method.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsCopy_32fc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit complex signed integer, vector copy method.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsCopy_64sc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit complex double, vector copy method.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsCopy_64fc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

    }
}