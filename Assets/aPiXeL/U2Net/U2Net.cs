using UnityEngine;
using Unity.Sentis;
using UnityEngine.Serialization;

public class U2Net : MonoBehaviour
{
    [SerializeField] public Texture2D _inputTexture;
    [SerializeField] public GameObject _inputObject;
    [SerializeField] public GameObject _outputObject;
    [SerializeField] public ModelAsset _backgroundRemover;
}
