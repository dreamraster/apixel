using UnityEngine;
using Unity.Sentis;
using Unity.Collections;
using Unity.Burst;
using System;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine.UIElements;

namespace aPiXeL
{
    // All Utilities
    [BurstCompile]
    internal class Utility
    {

        // Internal
        [BurstCompile]
        internal static unsafe void BurstAdd(float* a, float* b, float* c, int length)
        {
            for (int i = 0; i < length; ++i)
            {
                c[i] = a[i] + b[i];
            }
        }

        // Add Tensors
        internal static TensorFloat Add(TensorFloat a, TensorFloat b)
        {

            var roa = a.ToReadOnlyNativeArray();
            var rob = b.ToReadOnlyNativeArray();
            Debug.Assert(roa.Length.Equals(rob.Length));
            var res = new NativeArray<float>(roa.Length, Allocator.Temp);

            unsafe
            {
                var rawA = (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(roa);
                var rawB = (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(rob);
                var rawC = (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(res);
                BurstAdd(rawA, rawB, rawC, roa.Length);
            }
            return new TensorFloat(a.shape, res, 0);

        }

        // Internal
        [BurstCompile]
        internal static unsafe void BurstSubtract(float* a, float* b, float* c, int length)
        {
            for (int i = 0; i < length; ++i)
            {
                c[i] = a[i] - b[i];
            }
        }

        // Add Tensors
        internal static TensorFloat Subtract(TensorFloat a, TensorFloat b)
        {

            var roa = a.ToReadOnlyNativeArray();
            var rob = b.ToReadOnlyNativeArray();
            Debug.Assert(roa.Length.Equals(rob.Length));
            var res = new NativeArray<float>(roa.Length, Allocator.Temp);

            unsafe
            {
                var rawA = (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(roa);
                var rawB = (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(rob);
                var rawC = (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(res);
                BurstSubtract(rawA, rawB, rawC, roa.Length);
            }
            return new TensorFloat(a.shape, res, 0);

        }

        // Internal
        [BurstCompile]
        internal static unsafe void BurstMultiply(float* a, float b, float* c, int length)
        {
            for (int i = 0; i < length; ++i)
            {
                c[i] = a[i] * b;
            }
        }

        // Add Tensors
        internal static TensorFloat Multiply(TensorFloat a, float b)
        {

            var roa = a.ToReadOnlyNativeArray();
            var res = new NativeArray<float>(roa.Length, Allocator.Temp);

            unsafe
            {
                var rawA = (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(roa);
                var rawC = (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(res);
                BurstMultiply(rawA, b, rawC, roa.Length);
            }
            return new TensorFloat(a.shape, res, 0);

        }

        // Add Tensors
        internal static TensorFloat Divide(TensorFloat a, float b)
        {

            var roa = a.ToReadOnlyNativeArray();
            var res = new NativeArray<float>(roa.Length, Allocator.Temp);
            b = 1.0f / b;
            unsafe
            {
                var rawA = (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(roa);
                var rawC = (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(res);
                BurstMultiply(rawA, b, rawC, roa.Length);
            }
            return new TensorFloat(a.shape, res, 0);

        }

        // Burst Compile
        [BurstCompile]
        internal static unsafe void CreateRandom(uint seed, int length, float sigma, float* data)
        {
            Unity.Mathematics.Random random = Unity.Mathematics.Random.CreateFromIndex(seed);
            for (int i = 0; i < length; i++)
            {
                float u = random.NextFloat(0, 1.0f);
                float v = random.NextFloat(0, 1.0f);
                var radius = Math.Sqrt(-2.0 * Mathf.Log(u));
                var theta = 2.0 * Math.PI * v;
                var standardNormalRand = radius * Math.Cos(theta);
                data[i] = (float)standardNormalRand * sigma;
            }
        }

        // Create Random Tensor
        internal static TensorFloat CreateRandomTensor(int seed, int length, float sigma)
        {

            int _seed = (seed != int.MaxValue) ? seed : 2147483647;
            var res = new NativeArray<float>(length, Allocator.Temp);
            unsafe
            {
                var raw = (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(res);
                CreateRandom((uint)_seed, length, sigma, raw);
            }
            var shape = new TensorShape(new[] { length });
            return new TensorFloat(shape, res, 0);

        }

        // Create Random Tensor
        internal static TensorFloat CreateRandomTensor(int seed, TensorShape shape, float sigma)
        {

            int _seed = (seed != int.MaxValue) ? seed : 2147483647;
            var res = new NativeArray<float>(shape.length, Allocator.Temp);
            unsafe
            {
                var raw = (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(res);
                CreateRandom((uint)_seed, shape.length, sigma, raw);
            }
            return new TensorFloat(shape, res, 0);

        }

        [BurstCompile]
        internal static unsafe void Guidance(float* a, float* b, float* c, float guidance, int length)
        {
            for (int i = 0; i < length; ++i)
            {
                c[i] = a[i] + guidance * (b[i] - a[i]);
            }
        }

        // Add Tensors
        internal static TensorFloat Guidance(TensorFloat a, TensorFloat b, float c)
        {

            var roa = a.ToReadOnlyNativeArray();
            var rob = b.ToReadOnlyNativeArray();
            Debug.Assert(roa.Length.Equals(rob.Length));

            var res = new NativeArray<float>(roa.Length, Allocator.Temp);

            unsafe
            {
                var rawA = (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(roa);
                var rawB = (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(rob);
                var rawC = (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(res);
                Guidance(rawA, rawB, rawC, c, roa.Length);
            }

            return new TensorFloat(a.shape, res,0);

        }

    }

}
