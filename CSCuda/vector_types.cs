using System;

namespace CSCuda
{
    public struct float2
    {
        public float X;
        public float Y;

        public float2(string value)
        {
            var values = value.Split(',');

            X = Convert.ToSingle(values[0]);
            Y = Convert.ToSingle(values[1]);
        }

        public float2(float x, float y)
        {
            X = x;
            Y = y;
        }

        public float2(double x, double y)
        {
            X = Convert.ToSingle(x);
            Y = Convert.ToSingle(y);
        }

        public float2(int x, int y)
        {
            X = Convert.ToSingle(x);
            Y = Convert.ToSingle(y);
        }

        public override string ToString()
        {
            return string.Format("{0},{1}", X, Y);
        }
    }
}