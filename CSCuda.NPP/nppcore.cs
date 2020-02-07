using System;
using System.Runtime.InteropServices;
using CSCuda;

/**
 * \file nppcore.h
 * Basic NPP functionality.
 *  This file contains functions to query the NPP version as well as
 *  info about the CUDA compute capabilities on a given computer.
 */

namespace CSCuda.NPP
{
    /// <summary>
    /// Basic functions for library management, in particular library version
    /// and device property query functions.
    /// </summary>
    public static class NppCore
    {
        private const string dllFileName = @"nppc" + Constants.platform + "_" + Constants.major_version;  

        /// <summary>
        /// Get the NPP library version.
        /// </summary>
        /// <returns>A struct containing separate values for major and minor revision and build number.  </returns>
        public static NppLibraryVersion nppGetLibVersion()
        {
            IntPtr verPtr = NativeMethods.nppGetLibVersion();
            return (NppLibraryVersion)Marshal.PtrToStructure(verPtr, typeof(NppLibraryVersion));
        }

        /// <summary>
        /// What CUDA compute model is supported by the active CUDA device?
        ///
        /// Before trying to call any NPP functions, the user should make a call
        /// this function to ensure that the current machine has a CUDA capable device.
        /// </summary>
        /// <returns>An enum value representing if a CUDA capable device was found and what level of compute capabilities it supports.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppGpuComputeCapability nppGetGpuComputeCapability();

        /// <summary>
        /// Get the number of Streaming Multiprocessors (SM) on the active CUDA device.
        /// </summary>
        /// <returns>Number of SMs of the default CUDA device.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern int nppGetGpuNumSMs();

        /// <summary>
        /// Get the maximum number of threads per block on the active CUDA device.
        /// </summary>
        /// <returns>Maximum number of threads per block on the active CUDA device.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern int nppGetMaxThreadsPerBlock();

        /// <summary>
        /// Get the name of the active CUDA device.
        /// </summary>
        /// <returns>Name string of the active graphics-card/compute device in a system.</returns>
        public static String nppGetGpuName()
        {
            string deviceName = String.Empty;

            int bufferSize = 255;
            SByte[] devName = new SByte[bufferSize];
            IntPtr namePtr = NativeMethods.nppGetGpuName();

            deviceName = Marshal.PtrToStringAnsi(namePtr, bufferSize).TrimEnd('\0');

            return deviceName;
        }

        /// <summary>
        /// Get the maximum number of threads per SM for the active GPU
        /// </summary>
        /// <returns>Maximum number of threads per SM for the active GPU</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern int nppGetMaxThreadsPerSM();

        private static class NativeMethods
        {
            [DllImport(dllFileName, SetLastError = true)]
            public static extern IntPtr nppGetLibVersion();

            [DllImport(dllFileName, SetLastError = true)]
            public static extern IntPtr nppGetGpuName();
        }
    }
}