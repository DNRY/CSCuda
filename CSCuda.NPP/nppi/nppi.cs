using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CSCuda.NPP
{
    public partial class Nppi
    {
        private const string dllPrefix = @"nppi";
        private const string dllFileName = dllPrefix + Constants.platform + "_" + Constants.major_version;

        private const string arithLogic = "al";
        private const string alDll = dllPrefix + arithLogic + Constants.platform + "_" + Constants.major_version;

        private const string colorConversion = "cc";
        private const string ccDll = dllPrefix + colorConversion + Constants.platform + "_" + Constants.major_version;

        private const string compresssion = "com";
        private const string comDll = dllPrefix + compresssion + Constants.platform + "_" + Constants.major_version;
        
        private const string dataExchangeInit = "dei";
        private const string deiDll = dllPrefix + dataExchangeInit + Constants.platform + "_" + Constants.major_version;

        private const string filtering = "f";
        private const string fDll = dllPrefix + filtering + Constants.platform + "_" + Constants.major_version;

        private const string geometryTransform = "g";
        private const string gDll = dllPrefix + geometryTransform + Constants.platform + "_" + Constants.major_version;

        private const string morphologicalOperations = "m";
        private const string mDll = dllPrefix + morphologicalOperations + Constants.platform + "_" + Constants.major_version;

        private const string statisticsFunction = "st";
        private const string stDll = dllPrefix + statisticsFunction + Constants.platform + "_" + Constants.major_version;

        private const string supportFunctions = "su";
        private const string suDll = dllPrefix + supportFunctions + Constants.platform + "_" + Constants.major_version;

        private const string thresholdCompare = "tc";
        private const string tcDll = dllPrefix + thresholdCompare + Constants.platform + "_" + Constants.major_version;
    }
}
