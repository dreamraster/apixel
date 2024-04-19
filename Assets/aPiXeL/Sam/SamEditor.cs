using System;
using System.Collections.Generic;
using System.Linq;
using aPiXeL;
using Unity.Sentis;
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(Sam))]
public class SamEditor : Editor
{
    private static Engine SamEncodeEngine = null;
    private static Engine SamDecodeEngine = null;
    private static Vector2 ResizeRatio = Vector2.zero;
    private static RenderTexture SamTexture = null;
    private static Material PreProcessMaterial = null;
    
    Texture2D Resize(Texture2D texture2D,int targetX,int targetY)
    {
        RenderTexture rt=new RenderTexture(targetX, targetY,24);
        RenderTexture.active = rt;
        Graphics.Blit(texture2D,rt);
        Texture2D result=new Texture2D(targetX, targetY, TextureFormat.RGB24, false, true);
        result.ReadPixels(new Rect(0,0,targetX,targetY),0,0);
        result.Apply();
        return result;
    }
    
    private Texture PreProcess(Texture2D image)
    {
        var InputWidth = SamEncodeEngine.model.inputs[0].shape[3].value;
        var InputHeight = SamEncodeEngine.model.inputs[0].shape[2].value;
        SamTexture = RenderTexture.GetTemporary(InputWidth, InputHeight, 0, RenderTextureFormat.ARGBHalf);
        
        PreProcessMaterial = new Material(Shader.Find("BaseModel/PreProcess"));
        PreProcessMaterial.SetVector("_Mean", new Vector4(0.0f, 0.0f, 0.0f));
        PreProcessMaterial.SetVector("_Std", new Vector4(1.0f, 1.0f, 1.0f));
        PreProcessMaterial.SetFloat("_Max", 255.0f);  
        
        RenderTexture.active = SamTexture;
        Graphics.Blit(image, SamTexture, PreProcessMaterial);

        var result = new Texture2D(SamTexture.width, SamTexture.height, TextureFormat.RGBAHalf, false);
        result.ReadPixels(new Rect(0, 0, SamTexture.width, SamTexture.height), 0, 0);
        result.Apply();

        return result;
    }
    
    private Texture2D Square(Texture2D image)
    {
        var size = Math.Max(image.width, image.height);
        var SquareTexture = new Texture2D(size, size);
        SquareTexture.SetPixels(0, size - image.height, image.width, image.height, image.GetPixels());
        SquareTexture.Apply();
        return SquareTexture;
    }
    
    public Dictionary<string, TensorShape> GetInputShapes()
    {
        var InputShapes = new Dictionary<string, TensorShape>(SamEncodeEngine.model.inputs.Count);
        
        SamEncodeEngine.model.inputs.ForEach(input =>
        {
            var shape = input.shape.ToTensorShape();
            InputShapes[input.name] = shape;
        });

        return InputShapes;
    } 
    
    public Dictionary<string, Tensor> Predict(Texture2D image)
    {
        var InputShape = GetInputShapes().First().Value;
        var InputImage = PreProcess(image);
        var InputTensor = TextureConverter.ToTensor(InputImage, InputShape[3], InputShape[2], InputShape[1]);

        SamEncodeEngine.engine.Execute(InputTensor);

        var OutputTensor = new Dictionary<string, Tensor>(SamEncodeEngine.model.outputs.Count);
        SamEncodeEngine.model.outputs.ForEach(output => {
            var output_tensor = SamEncodeEngine.engine.PeekOutput(output.name);
            OutputTensor[output.name] = output_tensor;
        });

        InputTensor.Dispose();
        return OutputTensor;
    } 
    
    public TensorFloat Encode(Texture2D image)
    {
        var InputShape = GetInputShapes().First().Value;
        var Scale = InputShape[2] * (1.0f / Math.Max(image.width, image.height));
        var Width = (int)(image.width * Scale + 0.5f);
        var Height = (int)(image.height * Scale + 0.5f); 
        
        var ResizedTexture = Resize(image, Width, Height);
        var SquaredTexture = Square(ResizedTexture);

        ResizeRatio = new Vector2(
            (float)ResizedTexture.width / (float)image.width,
            (float)ResizedTexture.height / (float)image.height
        );

        var ImageEmbeds = Predict(SquaredTexture).First().Value as TensorFloat;

        MonoBehaviour.DestroyImmediate(ResizedTexture);
        MonoBehaviour.DestroyImmediate(SquaredTexture);

        return ImageEmbeds;
    } 

    public TensorFloat Decode(Texture2D Image, Tensor ImageEmbeds, List<Vector2> Points, List<float> Labels)
    {
        var _inputs = new Dictionary<string, Tensor>();
        _inputs.Add("image_embeddings", ImageEmbeds);

        var Coords = Points.SelectMany(point => new float[] { point.x, point.y }).ToArray();
        var PointCoords = new TensorFloat(new TensorShape(1, Points.Count, 2), Coords);
        _inputs.Add("point_coords", PointCoords);

        var PointLabels = new TensorFloat(new TensorShape(1, Points.Count), Labels.ToArray());
        _inputs.Add("point_labels", PointLabels);

        var MaskInputs = new TensorFloat(new TensorShape(1, 1, 256, 256), new float[256 * 256]);
        _inputs.Add("mask_input", MaskInputs);

        var HasMaskInput = new TensorFloat(new TensorShape(1), new float[] { 0.0f });
        _inputs.Add("has_mask_input", HasMaskInput);

        var OriginalImageSize = new TensorFloat(new TensorShape(2), new float[] { Image.height, Image.width });
        _inputs.Add("orig_im_size", OriginalImageSize);

        SamDecodeEngine.engine.Execute(_inputs);
        var masks = SamDecodeEngine.engine.PeekOutput("masks") as TensorFloat;
        return masks;
    }
    
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        Sam sd = (Sam)target;

        if (GUILayout.Button("Create"))
        {

            if (SamEncodeEngine == null)
            {
                SamEncodeEngine = new Engine();
                SamEncodeEngine.Init(sd._samEncoderAsset, BackendType.GPUCompute);
            }

            if (SamDecodeEngine == null)
            {
                SamDecodeEngine = new Engine();
                SamDecodeEngine.Init(sd._samDecoderAsset, BackendType.GPUCompute);
            }
            
            List<Vector2> points = new List<Vector2>();
            points.Add(sd.SegmentLocation);
            var labels = new List<float>() { 1.0f };
            var _input = Encode(sd._inputTexture);
            var _output = Decode(sd._inputTexture, _input, points, labels);
            
            // Create Color
            var renderTexture = TextureConverter.ToTexture(_output);
            RenderTexture.active = renderTexture;
            var texture = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false, true);
            texture.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
            texture.Apply();
            
            var _oututTexture = Resize(texture, sd._inputTexture.width, sd._inputTexture.height);
            byte[] bytes = ImageConversion.EncodeToPNG(_oututTexture);
            System.IO.File.WriteAllBytes("Assets/cleaned.png", bytes);
            AssetDatabase.ImportAsset("Assets/cleaned.png");
            AssetDatabase.Refresh();
            
            sd._inputObject.GetComponent<Renderer>().material.mainTexture = sd._inputTexture;
            sd._outputObject.GetComponent<Renderer>().material.mainTexture = AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/cleaned.png");
            
            _input.Dispose();
            _output.Dispose();
            
        }
        
    }
    
}