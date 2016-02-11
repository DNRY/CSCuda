using System;

/// file nppdefs.h
/// Typedefinitions and macros for NPP library.

namespace CSCuda.NPP
{
    using Npp16s = System.Int16;
    using Npp16u = System.UInt16;
    using Npp32f = System.Single;
    using Npp32s = System.Int32;
    using Npp32u = System.UInt32;
    using Npp64f = System.Double;
    using Npp64s = System.Int64;
    using Npp8u = System.Byte;
    using Npp8s = System.SByte;

    #region NPP Type Definitions and Constants

    /// <summary>
    /// Filtering methods.
    /// </summary>
    public enum NppiInterpolationMode
    {
        NPPI_INTER_UNDEFINED = 0,
        NPPI_INTER_NN = 1,        /**<  Nearest neighbor filtering. */
        NPPI_INTER_LINEAR = 2,        /**<  Linear interpolation. */
        NPPI_INTER_CUBIC = 4,        /**<  Cubic interpolation. */
        NPPI_INTER_CUBIC2P_BSPLINE,              /**<  Two-parameter cubic filter (B=1, C=0) */
        NPPI_INTER_CUBIC2P_CATMULLROM,           /**<  Two-parameter cubic filter (B=0, C=1/2) */
        NPPI_INTER_CUBIC2P_B05C03,               /**<  Two-parameter cubic filter (B=1/2, C=3/10) */
        NPPI_INTER_SUPER = 8,        /**<  Super sampling. */
        NPPI_INTER_LANCZOS = 16,       /**<  Lanczos filtering. */
        NPPI_INTER_LANCZOS3_ADVANCED = 17,       /**<  Generic Lanczos filtering with order 3. */
        NPPI_SMOOTH_EDGE = (1 << 31) /**<  Smooth edge filtering. */
    }

    /// <summary>
    /// Bayer Grid Position Registration.
    /// </summary>
    public enum NppiBayerGridPosition
    {
        NPPI_BAYER_BGGR = 0,             /**<  Default registration position. */
        NPPI_BAYER_RGGB = 1,
        NPPI_BAYER_GBRG = 2,
        NPPI_BAYER_GRBG = 3
    }

    /// <summary>
    /// Fixed filter-kernel sizes.
    /// </summary>
    public enum NppiMaskSize
    {
        NPP_MASK_SIZE_1_X_3,
        NPP_MASK_SIZE_1_X_5,
        NPP_MASK_SIZE_3_X_1 = 100, // leaving space for more 1 X N type enum values
        NPP_MASK_SIZE_5_X_1,
        NPP_MASK_SIZE_3_X_3 = 200, // leaving space for more N X 1 type enum values
        NPP_MASK_SIZE_5_X_5,
        NPP_MASK_SIZE_7_X_7 = 400,
        NPP_MASK_SIZE_9_X_9 = 500,
        NPP_MASK_SIZE_11_X_11 = 600,
        NPP_MASK_SIZE_13_X_13 = 700,
        NPP_MASK_SIZE_15_X_15 = 800
    }

    /// <summary>
    /// Error Status Codes
    ///
    /// Almost all NPP function return error-status information using
    /// these return codes.
    /// Negative return codes indicate errors, positive return codes indicate
    /// warnings, a return code of 0 indicates success.
    /// </summary>
    public enum NppStatus
    {
        /* negative return-codes indicate errors */
        NPP_NOT_SUPPORTED_MODE_ERROR = -9999,

        NPP_INVALID_HOST_POINTER_ERROR = -1032,
        NPP_INVALID_DEVICE_POINTER_ERROR = -1031,
        NPP_LUT_PALETTE_BITSIZE_ERROR = -1030,
        NPP_ZC_MODE_NOT_SUPPORTED_ERROR = -1028,      /**<  ZeroCrossing mode not supported  */
        NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY = -1027,
        NPP_TEXTURE_BIND_ERROR = -1024,
        NPP_WRONG_INTERSECTION_ROI_ERROR = -1020,
        NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR = -1006,
        NPP_MEMFREE_ERROR = -1005,
        NPP_MEMSET_ERROR = -1004,
        NPP_MEMCPY_ERROR = -1003,
        NPP_ALIGNMENT_ERROR = -1002,
        NPP_CUDA_KERNEL_EXECUTION_ERROR = -1000,

        NPP_ROUND_MODE_NOT_SUPPORTED_ERROR = -213,     /**< Unsupported round mode*/

        NPP_QUALITY_INDEX_ERROR = -210,     /**< Image pixels are constant for quality index */

        NPP_RESIZE_NO_OPERATION_ERROR = -201,     /**< One of the output image dimensions is less than 1 pixel */

        NPP_NOT_EVEN_STEP_ERROR = -108,     /**< Step value is not pixel multiple */
        NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR = -107,     /**< Number of levels for histogram is less than 2 */
        NPP_LUT_NUMBER_OF_LEVELS_ERROR = -106,     /**< Number of levels for LUT is less than 2 */

        NPP_CHANNEL_ORDER_ERROR = -60,      /**< Wrong order of the destination channels */
        NPP_ZERO_MASK_VALUE_ERROR = -59,      /**< All values of the mask are zero */
        NPP_QUADRANGLE_ERROR = -58,      /**< The quadrangle is nonconvex or degenerates into triangle, line or point */
        NPP_RECTANGLE_ERROR = -57,      /**< Size of the rectangle region is less than or equal to 1 */
        NPP_COEFFICIENT_ERROR = -56,      /**< Unallowable values of the transformation coefficients   */

        NPP_NUMBER_OF_CHANNELS_ERROR = -53,      /**< Bad or unsupported number of channels */
        NPP_COI_ERROR = -52,      /**< Channel of interest is not 1, 2, or 3 */
        NPP_DIVISOR_ERROR = -51,      /**< Divisor is equal to zero */

        NPP_CHANNEL_ERROR = -47,      /**< Illegal channel index */
        NPP_STRIDE_ERROR = -37,      /**< Stride is less than the row length */

        NPP_ANCHOR_ERROR = -34,      /**< Anchor point is outside mask */
        NPP_MASK_SIZE_ERROR = -33,      /**< Lower bound is larger than upper bound */

        NPP_RESIZE_FACTOR_ERROR = -23,
        NPP_INTERPOLATION_ERROR = -22,
        NPP_MIRROR_FLIP_ERROR = -21,
        NPP_MOMENT_00_ZERO_ERROR = -20,
        NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR = -19,
        NPP_THRESHOLD_ERROR = -18,
        NPP_CONTEXT_MATCH_ERROR = -17,
        NPP_FFT_FLAG_ERROR = -16,
        NPP_FFT_ORDER_ERROR = -15,
        NPP_STEP_ERROR = -14,       /**<  Step is less or equal zero */
        NPP_SCALE_RANGE_ERROR = -13,
        NPP_DATA_TYPE_ERROR = -12,
        NPP_OUT_OFF_RANGE_ERROR = -11,
        NPP_DIVIDE_BY_ZERO_ERROR = -10,
        NPP_MEMORY_ALLOCATION_ERR = -9,
        NPP_NULL_POINTER_ERROR = -8,
        NPP_RANGE_ERROR = -7,
        NPP_SIZE_ERROR = -6,
        NPP_BAD_ARGUMENT_ERROR = -5,
        NPP_NO_MEMORY_ERROR = -4,
        NPP_NOT_IMPLEMENTED_ERROR = -3,
        NPP_ERROR = -2,
        NPP_ERROR_RESERVED = -1,

        /* success */
        NPP_NO_ERROR = 0,        /**<  Error free operation */
        NPP_SUCCESS = NPP_NO_ERROR,                         /**<  Successful operation (same as NPP_NO_ERROR) */

        /* positive return-codes indicate warnings */
        NPP_NO_OPERATION_WARNING = 1,        /**<  Indicates that no operation was performed */
        NPP_DIVIDE_BY_ZERO_WARNING = 6,        /**<  Divisor is zero however does not terminate the execution */
        NPP_AFFINE_QUAD_INCORRECT_WARNING = 28,       /**<  Indicates that the quadrangle passed to one of affine warping functions doesn't have necessary properties. First 3 vertices are used, the fourth vertex discarded. */
        NPP_WRONG_INTERSECTION_ROI_WARNING = 29,       /**<  The given ROI has no interestion with either the source or destination ROI. Thus no operation was performed. */
        NPP_WRONG_INTERSECTION_QUAD_WARNING = 30,       /**<  The given quadrangle has no intersection with either the source or destination ROI. Thus no operation was performed. */
        NPP_DOUBLE_SIZE_WARNING = 35,       /**<  Image size isn't multiple of two. Indicates that in case of 422/411/420 sampling the ROI width/height was modified for proper processing. */

        NPP_MISALIGNED_DST_ROI_WARNING = 10000,    /**<  Speed reduction due to uncoalesced memory accesses warning. */
    }

    public enum NppGpuComputeCapability
    {
        NPP_CUDA_UNKNOWN_VERSION = -1,  /**<  Indicates that the compute-capability query failed */
        NPP_CUDA_NOT_CAPABLE = 0,   /**<  Indicates that no CUDA capable device was found */
        NPP_CUDA_1_0 = 100, /**<  Indicates that CUDA 1.0 capable device is machine's default device */
        NPP_CUDA_1_1 = 110, /**<  Indicates that CUDA 1.1 capable device is machine's default device */
        NPP_CUDA_1_2 = 120, /**<  Indicates that CUDA 1.2 capable device is machine's default device */
        NPP_CUDA_1_3 = 130, /**<  Indicates that CUDA 1.3 capable device is machine's default device */
        NPP_CUDA_2_0 = 200, /**<  Indicates that CUDA 2.0 capable device is machine's default device */
        NPP_CUDA_2_1 = 210, /**<  Indicates that CUDA 2.1 capable device is machine's default device */
        NPP_CUDA_3_0 = 300, /**<  Indicates that CUDA 3.0 capable device is machine's default device */
        NPP_CUDA_3_2 = 320, /**<  Indicates that CUDA 3.2 capable device is machine's default device */
        NPP_CUDA_3_5 = 350, /**<  Indicates that CUDA 3.5 capable device is machine's default device */
        NPP_CUDA_5_0 = 500  /**<  Indicates that CUDA 5.0 or better is machine's default device */
    }

    public struct NppLibraryVersion
    {
        public int major;   /**<  Major version number */
        public int minor;   /**<  Minor version number */
        public int build;   /**<  Build number. This reflects the nightly build this release was made from. */
    }

    #region Basic NPP Data Types

    /// <summary>
    /// Complex Number
    /// This struct represents an unsigned char complex number.
    /// </summary>
    public struct Npp8uc
    {
        public Npp8u re;     /**<  Real part */
        public Npp8u im;     /**<  Imaginary part */
    }

    /// <summary>
    /// Complex Number
    /// This struct represents a short complex number.
    /// </summary>
    public struct Npp16sc
    {
        public Npp16s re;     /**<  Real part */
        public Npp16s im;     /**<  Imaginary part */
    }

    /// <summary>
    /// Complex Number
    /// This struct represents an unsigned int complex number.
    /// </summary>
    public struct Npp32uc
    {
        public Npp32u re;     /**<  Real part */
        public Npp32u im;     /**<  Imaginary part */
    }

    /// <summary>
    /// Complex Number
    /// This struct represents a signed int complex number.
    /// </summary>
    public struct Npp32sc
    {
        public Npp32s re;     /**<  Real part */
        public Npp32s im;     /**<  Imaginary part */
    }

    /// <summary>
    /// Complex Number
    /// This struct represents a single floating-point complex number.
    /// </summary>
    public struct Npp32fc
    {
        public Npp32f re;     /**<  Real part */
        public Npp32f im;     /**<  Imaginary part */
    }

    /// <summary>
    /// Complex Number
    /// This struct represents a long long complex number.
    /// </summary>
    public struct Npp64sc
    {
        public Npp64s re;     /**<  Real part */
        public Npp64s im;     /**<  Imaginary part */
    }

    /// <summary>
    /// Complex Number
    /// This struct represents a double floating-point complex number.
    /// </summary>
    public struct Npp64fc
    {
        public Npp64f re;     /**<  Real part */
        public Npp64f im;     /**<  Imaginary part */
    }

    #endregion Basic NPP Data Types

    public static class NppConstants
    {
        public const byte NPP_MIN_8U = (0);                        /**<  Minimum 8-bit unsigned integer */
        public const byte NPP_MAX_8U = (255);                      /**<  Maximum 8-bit unsigned integer */
        public const ushort NPP_MIN_16U = (0);                        /**<  Minimum 16-bit unsigned integer */
        public const ushort NPP_MAX_16U = (65535);                    /**<  Maximum 16-bit unsigned integer */
        public const uint NPP_MIN_32U = (0);                        /**<  Minimum 32-bit unsigned integer */
        public const uint NPP_MAX_32U = (4294967295U);              /**<  Maximum 32-bit unsigned integer */
        public const ulong NPP_MIN_64U = (0);                        /**<  Minimum 64-bit unsigned integer */
        public const ulong NPP_MAX_64U = (18446744073709551615UL);  /**<  Maximum 64-bit unsigned integer */

        public const sbyte NPP_MIN_8S = (-127 - 1);                  /**<  Minimum 8-bit signed integer */
        public const sbyte NPP_MAX_8S = (127);                      /**<  Maximum 8-bit signed integer */
        public const short NPP_MIN_16S = (-32767 - 1);                /**<  Minimum 16-bit signed integer */
        public const short NPP_MAX_16S = (32767);                    /**<  Maximum 16-bit signed integer */
        public const int NPP_MIN_32S = (-2147483647 - 1);           /**<  Minimum 32-bit signed integer */
        public const int NPP_MAX_32S = (2147483647);               /**<  Maximum 32-bit signed integer */
        public const long NPP_MAX_64S = (9223372036854775807L);    /**<  Maximum 64-bit signed integer */
        public const long NPP_MIN_64S = (-9223372036854775807L - 1); /**<  Minimum 64-bit signed integer */

        public const float NPP_MINABS_32F = (1.175494351e-38f);         /**<  Smallest positive 32-bit floating point value */
        public const float NPP_MAXABS_32F = (3.402823466e+38f);         /**<  Largest  positive 32-bit floating point value */
        public const double NPP_MINABS_64F = (2.2250738585072014e-308);  /**<  Smallest positive 64-bit floating point value */
        public const double NPP_MAXABS_64F = (1.7976931348623158e+308);  /**<  Largest  positive 64-bit floating point value */
    }

    /// <summary>
    /// 2D Point
    /// </summary>
    public struct NppiPoint
    {
        public int x;      /**<  x-coordinate. */
        public int y;      /**<  y-coordinate. */
    }

    /// <summary>
    /// 2D Size
    /// This struct typically represents the size of a a rectangular region in
    /// two space.
    /// </summary>
    public struct NppiSize
    {
        public int width;  /**<  Rectangle width. */
        public int height; /**<  Rectangle height. */
    }

    /// <summary>
    /// 2D Rectangle
    /// This struct contains position and size information of a rectangle in
    /// two space.
    /// The rectangle's position is usually signified by the coordinate of its
    /// upper-left corner.
    /// </summary>
    public struct NppiRect
    {
        public int x;          /**<  x-coordinate of upper left corner. */
        public int y;          /**<  y-coordinate of upper left corner. */
        public int width;      /**<  Rectangle width. */
        public int height;     /**<  Rectangle height. */
    }

    public enum NppiAxis
    {
        NPP_HORIZONTAL_AXIS,
        NPP_VERTICAL_AXIS,
        NPP_BOTH_AXIS
    }

    public enum NppCmpOp
    {
        NPP_CMP_LESS,
        NPP_CMP_LESS_EQ,
        NPP_CMP_EQ,
        NPP_CMP_GREATER_EQ,
        NPP_CMP_GREATER
    }

    public enum NppRoundMode
    {
        /**
         * Round to the nearest even integer.
         * All fractional numbers are rounded to their nearest integer. The ambiguous
         * cases (i.e. \<integer\>.5) are rounded to the closest even integer.
         * E.g.
         * - roundNear(0.5) = 0
         * - roundNear(0.6) = 1
         * - roundNear(1.5) = 2
         * - roundNear(-1.5) = -2
         */
        NPP_RND_NEAR,
        NPP_ROUND_NEAREST_TIES_TO_EVEN = NPP_RND_NEAR, ///< Alias name for ::NPP_RND_NEAR.
        /**
         * Round according to financial rule.
         * All fractional numbers are rounded to their nearest integer. The ambiguous
         * cases (i.e. \<integer\>.5) are rounded away from zero.
         * E.g.
         * - roundFinancial(0.4)  = 0
         * - roundFinancial(0.5)  = 1
         * - roundFinancial(-1.5) = -2
         */
        NPP_RND_FINANCIAL,
        NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO = NPP_RND_FINANCIAL, ///< Alias name for ::NPP_RND_FINANCIAL.
        /**
         * Round towards zero (truncation).
         * All fractional numbers of the form \<integer\>.\<decimals\> are truncated to
         * \<integer\>.
         * - roundZero(1.5) = 1
         * - roundZero(1.9) = 1
         * - roundZero(-2.5) = -2
         */
        NPP_RND_ZERO,
        NPP_ROUND_TOWARD_ZERO = NPP_RND_ZERO, ///< Alias name for ::NPP_RND_ZERO.

        /*
         * Other rounding modes supported by IEEE-754 (2008) floating-point standard:
         *
         * - NPP_ROUND_TOWARD_INFINITY // ceiling
         * - NPP_ROUND_TOWARD_NEGATIVE_INFINITY // floor
         *
         */
    }

    public enum NppiBorderType
    {
        NPP_BORDER_UNDEFINED = 0,
        NPP_BORDER_NONE = NPP_BORDER_UNDEFINED,
        NPP_BORDER_CONSTANT = 1,
        NPP_BORDER_REPLICATE = 2,
        NPP_BORDER_WRAP = 3
    }

    public enum NppHintAlgorithm
    {
        NPP_ALG_HINT_NONE,
        NPP_ALG_HINT_FAST,
        NPP_ALG_HINT_ACCURATE
    }

    /* Alpha composition controls */

    public enum NppiAlphaOp
    {
        NPPI_OP_ALPHA_OVER,
        NPPI_OP_ALPHA_IN,
        NPPI_OP_ALPHA_OUT,
        NPPI_OP_ALPHA_ATOP,
        NPPI_OP_ALPHA_XOR,
        NPPI_OP_ALPHA_PLUS,
        NPPI_OP_ALPHA_OVER_PREMUL,
        NPPI_OP_ALPHA_IN_PREMUL,
        NPPI_OP_ALPHA_OUT_PREMUL,
        NPPI_OP_ALPHA_ATOP_PREMUL,
        NPPI_OP_ALPHA_XOR_PREMUL,
        NPPI_OP_ALPHA_PLUS_PREMUL,
        NPPI_OP_ALPHA_PREMUL
    }

    public struct NppiHaarClassifier_32f
    {
        public int numClassifiers;    /**<  number of classifiers */
        public IntPtr classifiers;       /**<  packed classifier data 40 bytes each */
        public ulong classifierStep;
        public NppiSize classifierSize;
        public IntPtr counterDevice;
    }

    public struct NppiHaarBuffer
    {
        public int haarBufferSize;    /**<  size of the buffer */
        public IntPtr haarBuffer;        /**<  buffer */
    }

    public enum NppsZCType
    {
        nppZCR,    /**<  sign change */
        nppZCXor,  /**<  sign change XOR */
        nppZCC     /**<  sign change count_0 */
    }

    public enum NppiHuffmanTableType
    {
        nppiDCTable,    /**<  DC Table */
        nppiACTable,    /**<  AC Table */
    }

    #endregion NPP Type Definitions and Constants
}