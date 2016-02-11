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
        /// Calculates the size of the temporary buffer for graph-cut with 4 neighborhood labeling.
        /// NOTE: This Graphcut function will be deprecated in a future release.
        /// </summary>
        /// <param name="oSize">Graph size.</param>
        /// <param name="pBufSize">Pointer to variable that returns the size of the temporary buffer.</param>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGraphcutGetSize(
            NppiSize oSize,
            IntPtr pBufSize);

        /// <summary>
        /// Calculates the size of the temporary buffer for graph-cut with 8 neighborhood labeling.
        /// NOTE: This Graphcut function will be deprecated in a future release.
        /// </summary>
        /// <param name="oSize">Graph size.</param>
        /// <param name="pBufSize">Pointer to variable that returns the size of the temporary buffer.</param>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGraphcut8GetSize(
            NppiSize oSize,
            IntPtr pBufSize);

        /// <summary>
        /// Initializes graph-cut state structure and allocates additional resources for graph-cut with 8 neighborhood labeling.
        /// NOTE: This Graphcut function will be deprecated in a future release.
        /// </summary>
        /// <param name="oSize">Graph size</param>
        /// <param name="ppState">Pointer to pointer to graph-cut state structure.</param>
        /// <param name="pDeviceMem">pDeviceMem to the sufficient amount of device memory. The CUDA runtime or NPP memory allocators must be used to allocate this memory. The minimum amount of device memory required to run graph-cut on a for a specific image size is computed by</param>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGraphcutInitAlloc(
            NppiSize oSize,
            IntPtr ppState,
            IntPtr pDeviceMem);

        /// <summary>
        /// Allocates and initializes the graph-cut state structure and additional resources for graph-cut with 8 neighborhood labeling.
        /// NOTE: This Graphcut function will be deprecated in a future release.
        /// </summary>
        /// <param name="oSize">Graph size</param>
        /// <param name="ppState">Pointer to pointer to graph-cut state structure.</param>
        /// <param name="pDeviceMem">to the sufficient amount of device memory. The CUDA runtime or NPP memory allocators must be used to allocate this memory. The minimum amount of device memory required to run graph-cut on a for a specific image size is computed by</param>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGraphcut8InitAlloc(
            NppiSize oSize,
            IntPtr ppState,
            IntPtr pDeviceMem);

        /// <summary>
        /// Frees the additional resources of the graph-cut state structure.
        /// NOTE: This Graphcut function will be deprecated in a future release.
        /// </summary>
        /// <param name="pState">Pointer to graph-cut state structure.</param>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGraphcutFree(
            IntPtr pState);

        /// <summary>
        /// Graphcut of a flow network (32bit signed integer edge capacities). The function computes the minimal cut (graphcut) of a 2D regular 4-connected graph. The inputs are the capacities of the horizontal (in transposed form), vertical and terminal (source and sink) edges. The capacities to source and sink are stored as capacity differences in the terminals array ( terminals(x) = source(x) - sink(x) ). The implementation assumes that the edge capacities for boundary edges that would connect to nodes outside the specified domain are set to 0 (for example left(0,*) == 0). If this is not fulfilled the computed labeling may be wrong! The computed binary labeling is encoded as unsigned 8bit values (0 and >0).
        /// NOTE: This Graphcut function will be deprecated in a future release.
        /// </summary>
        /// <param name="pTerminals">Pointer to differences of terminal edge capacities (terminal(x) = source(x) - sink(x))</param>
        /// <param name="pLeftTransposed">Pointer to transposed left edge capacities (left(0,*) must be 0)</param>
        /// <param name="pRightTransposed">Pointer to transposed right edge capacities (right(width-1,*) must be 0)</param>
        /// <param name="pTop">Pointer to top edge capacities (top(*,0) must be 0)</param>
        /// <param name="pBottom">Pointer to bottom edge capacities (bottom(*,height-1) must be 0)</param>
        /// <param name="nStep">Step in bytes between any pair of sequential rows of edge capacities</param>
        /// <param name="nTransposedStep">Step in bytes between any pair of sequential rows of tranposed edge capacities</param>
        /// <param name="size">Graph size</param>
        /// <param name="pLabel">Pointer to destination label image</param>
        /// <param name="nLabelStep">Step in bytes between any pair of sequential rows of label image</param>
        /// <param name="pState">Pointer to graph-cut state structure. This structure must be initialized allocated and initialized using</param>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGraphcut_32s8u(
            IntPtr pTerminals,
            IntPtr pLeftTransposed,
            IntPtr pRightTransposed,
            IntPtr pTop,
            IntPtr pBottom,
            int nStep,
            int nTransposedStep,
            NppiSize size,
            IntPtr pLabel,
            int nLabelStep,
            IntPtr pState);

        /// <summary>
        /// Graphcut of a flow network (32bit signed integer edge capacities). The function computes the minimal cut (graphcut) of a 2D regular 8-connected graph. The inputs are the capacities of the horizontal (in transposed form), vertical and terminal (source and sink) edges. The capacities to source and sink are stored as capacity differences in the terminals array ( terminals(x) = source(x) - sink(x) ). The implementation assumes that the edge capacities for boundary edges that would connect to nodes outside the specified domain are set to 0 (for example left(0,*) == 0). If this is not fulfilled the computed labeling may be wrong! The computed binary labeling is encoded as unsigned 8bit values (0 and >0).
        /// NOTE: This Graphcut function will be deprecated in a future release.
        /// </summary>
        /// <param name="pTerminals">Pointer to differences of terminal edge capacities (terminal(x) = source(x) - sink(x))</param>
        /// <param name="pLeftTransposed">Pointer to transposed left edge capacities (left(0,*) must be 0)</param>
        /// <param name="pRightTransposed">Pointer to transposed right edge capacities (right(width-1,*) must be 0)</param>
        /// <param name="pTop">Pointer to top edge capacities (top(*,0) must be 0)</param>
        /// <param name="pTopLeft">Pointer to top left edge capacities (topleft(*,0) & topleft(0,*) must be 0)</param>
        /// <param name="pTopRight">Pointer to top right edge capacities (topright(*,0) & topright(width-1,*) must be 0)</param>
        /// <param name="pBottom">Pointer to bottom edge capacities (bottom(*,height-1) must be 0)</param>
        /// <param name="pBottomLeft">Pointer to bottom left edge capacities (bottomleft(*,height-1) && bottomleft(0,*) must be 0)</param>
        /// <param name="pBottomRight">Pointer to bottom right edge capacities (bottomright(*,height-1) && bottomright(width-1,*) must be 0)</param>
        /// <param name="nStep">Step in bytes between any pair of sequential rows of edge capacities</param>
        /// <param name="nTransposedStep">Step in bytes between any pair of sequential rows of tranposed edge capacities</param>
        /// <param name="size">Graph size</param>
        /// <param name="pLabel">Pointer to destination label image</param>
        /// <param name="nLabelStep">Step in bytes between any pair of sequential rows of label image</param>
        /// <param name="pState">Pointer to graph-cut state structure. This structure must be initialized allocated and initialized using</param>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGraphcut8_32s8u(
            IntPtr pTerminals,
            IntPtr pLeftTransposed,
            IntPtr pRightTransposed,
            IntPtr pTop,
            IntPtr pTopLeft,
            IntPtr pTopRight,
            IntPtr pBottom,
            IntPtr pBottomLeft,
            IntPtr pBottomRight,
            int nStep,
            int nTransposedStep,
            NppiSize size,
            IntPtr pLabel,
            int nLabelStep,
            IntPtr pState);

        /// <summary>
        /// Graphcut of a flow network (32bit float edge capacities). The function computes the minimal cut (graphcut) of a 2D regular 4-connected graph. The inputs are the capacities of the horizontal (in transposed form), vertical and terminal (source and sink) edges. The capacities to source and sink are stored as capacity differences in the terminals array ( terminals(x) = source(x) - sink(x) ). The implementation assumes that the edge capacities for boundary edges that would connect to nodes outside the specified domain are set to 0 (for example left(0,*) == 0). If this is not fulfilled the computed labeling may be wrong! The computed binary labeling is encoded as unsigned 8bit values (0 and >0).
        /// NOTE: This Graphcut function will be deprecated in a future release.
        /// </summary>
        /// <param name="pTerminals">Pointer to differences of terminal edge capacities (terminal(x) = source(x) - sink(x))</param>
        /// <param name="pLeftTransposed">Pointer to transposed left edge capacities (left(0,*) must be 0)</param>
        /// <param name="pRightTransposed">Pointer to transposed right edge capacities (right(width-1,*) must be 0)</param>
        /// <param name="pTop">Pointer to top edge capacities (top(*,0) must be 0)</param>
        /// <param name="pBottom">Pointer to bottom edge capacities (bottom(*,height-1) must be 0)</param>
        /// <param name="nStep">Step in bytes between any pair of sequential rows of edge capacities</param>
        /// <param name="nTransposedStep">Step in bytes between any pair of sequential rows of tranposed edge capacities</param>
        /// <param name="size">Graph size</param>
        /// <param name="pLabel">Pointer to destination label image</param>
        /// <param name="nLabelStep">Step in bytes between any pair of sequential rows of label image</param>
        /// <param name="pState">Pointer to graph-cut state structure. This structure must be initialized allocated and initialized using</param>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGraphcut_32f8u(
            IntPtr pTerminals,
            IntPtr pLeftTransposed,
            IntPtr pRightTransposed,
            IntPtr pTop,
            IntPtr pBottom,
            int nStep,
            int nTransposedStep,
            NppiSize size,
            IntPtr pLabel,
            int nLabelStep,
            IntPtr pState);

        /// <summary>
        /// Graphcut of a flow network (32bit float edge capacities). The function computes the minimal cut (graphcut) of a 2D regular 8-connected graph. The inputs are the capacities of the horizontal (in transposed form), vertical and terminal (source and sink) edges. The capacities to source and sink are stored as capacity differences in the terminals array ( terminals(x) = source(x) - sink(x) ). The implementation assumes that the edge capacities for boundary edges that would connect to nodes outside the specified domain are set to 0 (for example left(0,*) == 0). If this is not fulfilled the computed labeling may be wrong! The computed binary labeling is encoded as unsigned 8bit values (0 and >0).
        /// NOTE: This Graphcut function will be deprecated in a future release.
        /// </summary>
        /// <param name="pTerminals">Pointer to differences of terminal edge capacities (terminal(x) = source(x) - sink(x))</param>
        /// <param name="pLeftTransposed">Pointer to transposed left edge capacities (left(0,*) must be 0)</param>
        /// <param name="pRightTransposed">Pointer to transposed right edge capacities (right(width-1,*) must be 0)</param>
        /// <param name="pTop">Pointer to top edge capacities (top(*,0) must be 0)</param>
        /// <param name="pTopLeft">Pointer to top left edge capacities (topleft(*,0) & topleft(0,*) must be 0)</param>
        /// <param name="pTopRight">Pointer to top right edge capacities (topright(*,0) & topright(width-1,*) must be 0)</param>
        /// <param name="pBottom">Pointer to bottom edge capacities (bottom(*,height-1) must be 0)</param>
        /// <param name="pBottomLeft">Pointer to bottom left edge capacities (bottomleft(*,height-1) && bottomleft(0,*) must be 0)</param>
        /// <param name="pBottomRight">Pointer to bottom right edge capacities (bottomright(*,height-1) && bottomright(width-1,*) must be 0)</param>
        /// <param name="nStep">Step in bytes between any pair of sequential rows of edge capacities</param>
        /// <param name="nTransposedStep">Step in bytes between any pair of sequential rows of tranposed edge capacities</param>
        /// <param name="size">Graph size</param>
        /// <param name="pLabel">Pointer to destination label image</param>
        /// <param name="nLabelStep">Step in bytes between any pair of sequential rows of label image</param>
        /// <param name="pState">Pointer to graph-cut state structure. This structure must be initialized allocated and initialized using</param>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGraphcut8_32f8u(
            IntPtr pTerminals,
            IntPtr pLeftTransposed,
            IntPtr pRightTransposed,
            IntPtr pTop,
            IntPtr pTopLeft,
            IntPtr pTopRight,
            IntPtr pBottom,
            IntPtr pBottomLeft,
            IntPtr pBottomRight,
            int nStep,
            int nTransposedStep,
            NppiSize size,
            IntPtr pLabel,
            int nLabelStep,
            IntPtr pState);

    }
}