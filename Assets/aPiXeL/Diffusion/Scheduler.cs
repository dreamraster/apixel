using NumSharp;
using System.Collections.Generic;
using System.Linq;
using System;
using Unity.Sentis;
using UnityEngine;

namespace aPiXeL
{
    public class Scheduler
    {

        private bool _is_scale_input_called;
        private readonly int _num_steps;
        private List<float> _alphas_cumulative_products;

        public TensorFloat Sigmas { get; set; }
        public List<int> Timesteps { get; set; }
        public float InitNoiseSigma { get; set; }

        public Scheduler()
        {
            _num_steps = 1000;

            float beta_start = 0.00085f;
            float beta_end = 0.012f;

            var betas = Enumerable.Range(0, _num_steps).Select(i => beta_start + (beta_end - beta_start) * i / (_num_steps - 1)).ToList();
            var alphas = betas.Select(beta => 1 - beta).ToList();
            this._alphas_cumulative_products = alphas.Select((alpha, i) => alphas.Take(i + 1).Aggregate((a, b) => a * b)).ToList();
            var sigmas = _alphas_cumulative_products.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();
            this.InitNoiseSigma = (float)sigmas.Max();
        }

        public static double[] Interpolate(double[] timesteps, double[] range, List<double> sigmas)
        {

            var result = np.zeros(timesteps.Length + 1);
            for (int i = 0; i < timesteps.Length; i++)
            {
                int index = Array.BinarySearch(range, timesteps[i]);

                if (index >= 0)
                {
                    result[i] = sigmas[index];
                }
                else if (index == -1)
                {
                    result[i] = sigmas[0];
                }
                else if (index == -range.Length - 1)
                {
                    result[i] = sigmas[range.Length - 1];
                }
                else
                {
                    index = ~index;
                    double t = (timesteps[i] - range[index - 1]) / (range[index] - range[index - 1]);
                    result[i] = sigmas[index - 1] + t * (sigmas[index] - sigmas[index - 1]);
                }
            }
            result = np.add(result, 0.000f);
            return result.ToArray<double>();

        }

        public int[] SetTimesteps(int num_inference_steps)
        {

            double start = 0;
            double stop = _num_steps - 1;
            double[] timesteps = np.linspace(start, stop, num_inference_steps).ToArray<double>();

            this.Timesteps = timesteps.Select(x => (int)x).Reverse().ToList();

            var sigmas = _alphas_cumulative_products.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();
            var range = np.arange((double)0, (double)(sigmas.Count)).ToArray<double>();
            sigmas = Interpolate(timesteps, range, sigmas).ToList();
            this.InitNoiseSigma = (float)sigmas.Max();

            var shape = new TensorShape(sigmas.Count());
            var sigma = new float[sigmas.Count()];
            for (int i = 0; i < sigmas.Count(); i++)
                sigma[i] = (float)sigmas[i];
            this.Sigmas = new TensorFloat(shape, sigma);

            return this.Timesteps.ToArray();

        }

        public TensorFloat Step(TensorFloat modelOutput, int timestep, TensorFloat sample, int seed)
        {

            Debug.Assert(this._is_scale_input_called);

            var stepIndex = this.Timesteps.IndexOf((int)timestep);
            var sigma = this.Sigmas[stepIndex];

            modelOutput.MakeReadable();
            var moduleOutputSample = Utility.Multiply(modelOutput, sigma);
            sample.MakeReadable();
            moduleOutputSample.MakeReadable();
            var predOriginalSample = Utility.Subtract(sample, moduleOutputSample);

            var sigmaFrom = this.Sigmas[stepIndex];
            var sigmaTo = this.Sigmas[stepIndex + 1];

            var sigmaFromLessSigmaTo = (MathF.Pow(sigmaFrom, 2) - MathF.Pow(sigmaTo, 2));
            var sigmaUpResult = (MathF.Pow(sigmaTo, 2) * sigmaFromLessSigmaTo) / MathF.Pow(sigmaFrom, 2);
            var sigmaUp = sigmaUpResult < 0 ? -MathF.Pow(MathF.Abs(sigmaUpResult), 0.5f) : MathF.Pow(sigmaUpResult, 0.5f);

            var sigmaDownResult = (MathF.Pow(sigmaTo, 2) - MathF.Pow(sigmaUp, 2));
            var sigmaDown = sigmaDownResult < 0 ? -MathF.Pow(MathF.Abs(sigmaDownResult), 0.5f) : MathF.Pow(sigmaDownResult, 0.5f);

            predOriginalSample.MakeReadable();
            var sampleMinusPredOriginalSample = Utility.Subtract(sample, predOriginalSample);
            sampleMinusPredOriginalSample.MakeReadable();
            var derivative = Utility.Divide(sampleMinusPredOriginalSample, sigma);

            var dt = sigmaDown - sigma;

            derivative.MakeReadable();
            var derivativeSample = Utility.Multiply(derivative, dt);
            derivativeSample.MakeReadable();
            var prevSample = Utility.Add(sample, derivativeSample);

            var noise = Utility.CreateRandomTensor(seed, prevSample.shape, 1.0f);
            noise.MakeReadable();

            var noiseSigmaUpProduct = Utility.Multiply(noise, sigmaUp);
            prevSample.MakeReadable();
            noiseSigmaUpProduct.MakeReadable();
            prevSample = Utility.Add(prevSample, noiseSigmaUpProduct);
            return prevSample;

        }

        public TensorFloat ScaleInput(TensorFloat sample, int timestep)
        {
            int stepIndex = this.Timesteps.IndexOf(timestep);
            var sigma = this.Sigmas[stepIndex];
            sigma = (float)Math.Sqrt((Math.Pow(sigma, 2) + 1));

            sample.MakeReadable();
            sample = Utility.Divide(sample, sigma);
            _is_scale_input_called = true;
            return sample;
        }
    }
}
