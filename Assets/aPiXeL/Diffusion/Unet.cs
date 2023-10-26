using Unity.Sentis;
using System.Collections.Generic;

namespace aPiXeL
{

    // Execution Engine.
    internal class Unet : Engine
    {
        public static Dictionary<string, Unity.Sentis.Tensor> Prepare(TensorFloat text, TensorFloat sample, int timeStep)
        {
            var shape = new TensorShape(1);
            var inputTensor = new TensorInt(shape, new int[] { timeStep });
            Dictionary<string, Unity.Sentis.Tensor> inputTensors = new Dictionary<string, Unity.Sentis.Tensor>() {
                { "encoder_hidden_states", text },
                { "sample", sample },
                { "timestep", inputTensor },
            };
            return inputTensors;
        }

        internal TensorFloat Do(TensorFloat text, TensorFloat sample, int timeStep)
        {
            // Do all input output validation here
            var input = Prepare(text, sample, timeStep);
            return Execute(input) as TensorFloat;
        }
    };

}
