using Unity.Sentis;

namespace aPiXeL
{

    // Execution Engine.
    internal class Tokenizer : Engine
    {
        internal TensorInt Do(TensorInt input)
        {
            // Do all input output validation here
            // Convert String to Int somehow.
            return Execute(input) as TensorInt;
        }
    };

}
