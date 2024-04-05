using aPiXeL;
using Unity.Collections;
using Unity.Sentis;
using UnityEngine;
using UnityEditor;
using UnityEditor.VersionControl;

[CustomEditor(typeof(U2Net))]
public class U2NetEditor : Editor
{
    private static Engine U2NetEngine = null;
    
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
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        U2Net sd = (U2Net)target;

        if (GUILayout.Button("Create"))
        {

            if (U2NetEngine == null)
            {
                U2NetEngine = new Engine();
                U2NetEngine.Init(sd._backgrundRemover, BackendType.CPU);
            }

            var _input = Resize(sd._inputTexture, 320, 320);
            var _texture = TextureConverter.ToTensor(_input);
            var _output = U2NetEngine.Execute(_texture) as TensorFloat;
            _output.MakeReadable(); 
            
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
            
            _texture.Dispose();
            _output.Dispose();
            
        }
        
    }
    
}