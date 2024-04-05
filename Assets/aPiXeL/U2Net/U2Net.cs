using UnityEngine;
using Unity.Sentis;
public class U2Net : MonoBehaviour
{
    [SerializeField] public Texture2D _inputTexture;
    [SerializeField] public GameObject _inputObject;
    [SerializeField] public GameObject _outputObject;
    [SerializeField] public ModelAsset _backgrundRemover;
}
