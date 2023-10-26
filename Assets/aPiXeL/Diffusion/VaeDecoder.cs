using Unity.Sentis;

namespace aPiXeL
{

    // Execution Engine.
    internal class VaeDecoder : Engine
    {
        internal TensorFloat Do(TensorFloat input)
        {
            // Do all input output validation here
            return Execute(input) as TensorFloat;
        }
    };

}
