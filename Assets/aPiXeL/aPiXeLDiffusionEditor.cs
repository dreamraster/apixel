using aPiXeL;
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(aPiXeLDiffusion))]
public class aPiXeLDiffusionEditor : Editor
{

    static Diffuser diffuser = null;
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        aPiXeLDiffusion sd = (aPiXeLDiffusion)target;

        if (GUILayout.Button("Create"))
        {
            if (diffuser == null)
            {
                diffuser = new Diffuser();
                diffuser.Initialize(sd._unetModel, sd._textEncoderModel, sd._vaeModel, null);
            }

            var path = Application.dataPath + "/" + sd.Prompt() + sd.Seed()+ ".png";
            diffuser.Execute(sd.Prompt(), sd.Steps(), 8.0f, sd.Seed(), path);

            AssetDatabase.Refresh();

            path = "Assets/" + sd.Prompt() + sd.Seed() + ".png";
            var texture = AssetDatabase.LoadAssetAtPath<Texture2D>(path);
            path = "Assets/" + sd.Prompt() + sd.Seed() + ".mat";
            var material = new Material(Shader.Find("Standard"));
            material.SetTexture("_MainTex", texture);
            AssetDatabase.CreateAsset(material, path);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();

            sd.Target().GetComponent<MeshRenderer>().sharedMaterial = AssetDatabase.LoadAssetAtPath<Material>(path);
            AssetDatabase.Refresh();
        }
    }
}
