using System.Linq;
using Unity.Sentis;

namespace aPiXeL
{

    // Execution Engine.
    internal class TextEncoder : Engine
    {

        private const int _max_length = 77;
        private const int _blank_token = 49407;
        private const int _embedding_size = 59136;   // 77 * 768    (Check Text Model for SD 1/2)

        internal TensorFloat Do(TensorInt input)
        {

            // Text Encoder | Generate for Blank Inputs.
            var textInput = Enumerable.Repeat(_blank_token - 1, _max_length).ToArray();
            var textShape = new TensorShape(1, _max_length);
            var textInputTensor = new TensorInt(textShape, textInput);

            var textOutputTensor = Execute(textInputTensor) as TensorFloat;

            var results = new float[2 * _embedding_size];
            for (int i = 0; i < _embedding_size; ++i)
                results[i] = textOutputTensor[i];

            // Text Encoder | Generate for Prompt.
            for (int i = 0; i < input.shape[0]; ++i)
                textInputTensor[i] = input[i];
            textOutputTensor = Execute(textInputTensor) as TensorFloat;

            for (int i = 0; i < _embedding_size; ++i)
                results[_embedding_size + i] = textOutputTensor[i];
            textInputTensor.Dispose();
            textOutputTensor.Dispose();

            var outputShape = new TensorShape(2 * _embedding_size);
            var outputTensor = new TensorFloat(outputShape, results);
            return outputTensor;

        }
    };

}
