using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Unity.Sentis;
using UnityEngine;
using Unity.Collections;
using UnityEngine.UIElements;
using Unity.Collections.LowLevel.Unsafe;

namespace aPiXeL
{

    internal class TokenizerONNX : Engine
    {

        private InferenceSession session;

        internal void Init(string path, string extension)
        {
            var sessionOptions = new SessionOptions();
            sessionOptions.RegisterCustomOpLibraryV2(extension, out _);
            session = new InferenceSession(path, sessionOptions);
        }

        internal override void Destroy()
        {
            session.Dispose();
        }

        internal int[] Execute(string text)
        {

            var inputTensor = new DenseTensor<string>(new string[] { text }, new int[] { 1 });
            var inputString = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<string>("string_input", inputTensor) };

            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> tokens = null;
            if (session != null)
                tokens = session.Run(inputString);

            var inputIds = (tokens.ToList().First().Value as IEnumerable<long>).ToArray();
            var inputIdsInt = inputIds.Select(x => (int)x).ToArray();
            return inputIdsInt;
        }

    }

    // Execution Engine.
    public class Diffuser
    {

        private Unet _unet = new Unet();
        private TextEncoder _text_encoder = new TextEncoder();
        private VaeDecoder _vae = new VaeDecoder();
        private TokenizerONNX _text_tokenizer = new TokenizerONNX();
        private Scheduler _scheduler = new Scheduler();

        private const float _scale = 1.0f / 0.18215f;

        public void Initialize(ModelAsset unet, ModelAsset text, ModelAsset vae, ModelAsset tokenizer)
        {
            _unet.Init(unet, BackendType.GPUCompute);
            _text_encoder.Init(text, BackendType.GPUCompute);
            _vae.Init(vae, BackendType.CPU);

            var _tokenizerModePath = Application.dataPath + $"/Models/sd_14/tokenizer/cliptokenizer.onnx";
            var _ortextensionsPath = Application.dataPath + $"/Plugins/ortextensions.dll";
            _text_tokenizer.Init(_tokenizerModePath, _ortextensionsPath);
        }

        private static TensorFloat Guidance(TensorFloat noisePred, TensorFloat noisePredText, float guidanceScale)
        {
            noisePred = Utility.Guidance(noisePred, noisePredText, guidanceScale);
            return noisePred;
        }

        private static Tuple<TensorFloat, TensorFloat> Split(TensorFloat tensorToSplit, GenerationSettings settings)
        {
            var dims = new int[4] { settings.BatchSize, settings.Channels, settings.Width, settings.Height };
            var shape = new TensorShape(dims);
            var length = shape.length;
            var tensor1Array = new NativeArray<float>(length, Allocator.Temp);
            var tensor2Array = new NativeArray<float>(length, Allocator.Temp);

            var tensorToSplitArray = tensorToSplit.ToReadOnlyNativeArray();
            NativeArray<float>.Copy(tensorToSplitArray, 0, tensor1Array, 0, length);
            NativeArray<float>.Copy(tensorToSplitArray, length, tensor2Array, 0, length);

            var tensor1 = new TensorFloat(shape, tensor1Array, 0);
            var tensor2 = new TensorFloat(shape, tensor2Array, 0);
            return new Tuple<TensorFloat, TensorFloat>(tensor1, tensor2);
        }

        private static TensorFloat GenerateLatent(int seed, float noiseSigma, GenerationSettings settings)
        {
            var shape = new TensorShape(new[] { settings.BatchSize, settings.Channels, settings.Width, settings.Height });
            var _seed = (seed != int.MaxValue) ? seed : 2147483647;
            var _size = settings.BatchSize * settings.Channels * settings.Width * settings.Height;
            var result = new NativeArray<float>(_size, Allocator.Temp);
            unsafe
            {
                var raw = (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(result);
                Utility.CreateRandom((uint)_seed, shape.length, noiseSigma, raw);
            }
            return new TensorFloat(shape, result, 0);
        }

        public void Execute(string prompt, int steps, float guidance, int seed, string outputPath)
        {

            // Settings.
            GenerationSettings settings = new GenerationSettings();
            settings.BatchSize = 1;
            settings.Channels = 4;
            settings.Width = 512 / 8;
            settings.Height = 512 / 8;

            // Text Tokenizer
            var tokenizedText = _text_tokenizer.Execute(prompt);
            var textShape = new TensorShape(tokenizedText.Length);
            var textTensor = new TensorInt(textShape, tokenizedText);

            // Text Encoder.
            var textEncoder = _text_encoder.Do(textTensor);
            var textEmbeddingShape = new TensorShape(2, 77, 768);
            var textEmbeddingTensor = new TensorFloat(textEmbeddingShape, textEncoder.ToReadOnlyArray());
            var timeStart = Time.realtimeSinceStartup;

            // Unet Inference.
            var timeSteps = _scheduler.SetTimesteps(steps);
            var latents = GenerateLatent(4, _scheduler.InitNoiseSigma, settings);
            var random = Unity.Mathematics.Random.CreateFromIndex((uint)seed);
            Debug.LogFormat("Text Encoder {0} seconds elapsed", Time.realtimeSinceStartup - timeStart);

            for (int t = 0; t < steps; t++)
            {
                var latentData = latents.ToReadOnlyArray();
                var latentShape = new TensorShape(new[] { settings.BatchSize * 2, settings.Channels, settings.Width, settings.Height });
                var latentDouble = latentData.Concat(latentData).ToArray();
                var latentInput = new TensorFloat(latentShape, latentDouble);

                latentInput = _scheduler.ScaleInput(latentInput, timeSteps[t]);
                var textOutputTensor = _unet.Do(textEmbeddingTensor, latentInput, timeSteps[t]) as TensorFloat;

                var splitTensors = Split(textOutputTensor, settings);
                var noisePred = splitTensors.Item1;
                var noisePredText = splitTensors.Item2;
                noisePred = Guidance(noisePred, noisePredText, guidance);

                latents = _scheduler.Step(noisePred, timeSteps[t], latents, random.NextInt());

                Debug.LogFormat("After {0} steps {1} seconds elapsed", t, Time.realtimeSinceStartup - timeStart);
            }

            var latentsScaled = Utility.Multiply(latents, _scale);
            var decoderOutput = _vae.Execute(latentsScaled) as TensorFloat;

            Debug.LogFormat("Inference completed in {0} seconds", Time.realtimeSinceStartup - timeStart);

            // Encode texture into PNG
            var renderTexture = TextureConverter.ToTexture(decoderOutput);
            RenderTexture.active = renderTexture;
            var texture = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false, true);
            texture.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
            texture.Apply();
            byte[] bytes = ImageConversion.EncodeToPNG(texture);
            File.WriteAllBytes(outputPath, bytes);
        }

        public void Destroy()
        {

            _text_tokenizer.Destroy();
            _vae.Destroy();
            _text_encoder.Destroy();
            _unet.Destroy();

        }

    };

}
