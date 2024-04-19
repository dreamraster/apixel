using UnityEngine;
using Unity.Sentis;
using System.Collections;
using System.Collections.Generic;

namespace aPiXeL
{

    // Execution Engine.
    internal class Engine
    {

        internal Model model;
        internal IWorker engine;

        internal void Init(ModelAsset modelAsset, BackendType backend = BackendType.GPUCompute)
        {
            model = ModelLoader.Load(modelAsset);
            engine = WorkerFactory.CreateWorker(backend, model);
        }

        internal void DebugInputOutput()
        {
            foreach (var input in model.inputs)
                Debug.LogFormat("{0} - {1}", input.name, input.shape);
            foreach (var output in model.outputs)
                Debug.Log(output);
        }

        internal Tensor ExecuteStep(IDictionary<string, Unity.Sentis.Tensor> input)
        {
            bool progress = false;

            IEnumerator modelEnumerator = engine.ExecuteLayerByLayer(input);
            do
            {
                progress = modelEnumerator.MoveNext();
            }
            while (progress);

            return engine.PeekOutput();
        }

        internal Unity.Sentis.Tensor Execute(IDictionary<string, Unity.Sentis.Tensor> input)
        {
            return engine.Execute(input).PeekOutput();
        }

        internal Unity.Sentis.Tensor ExecuteStep(Unity.Sentis.Tensor input)
        {
            bool progress = false;

            IEnumerator modelEnumerator = engine.ExecuteLayerByLayer(input);
            do
            {
                progress = modelEnumerator.MoveNext();
            }
            while (progress);

            return engine.PeekOutput();
        }

        internal Unity.Sentis.Tensor Execute(Unity.Sentis.Tensor input)
        {
            return engine.Execute(input).PeekOutput();
        }

        virtual internal void Destroy()
        {
            engine.Dispose();
        }

    }

}
