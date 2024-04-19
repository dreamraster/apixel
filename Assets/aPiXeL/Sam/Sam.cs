using UnityEngine;
using Unity.Sentis;
using UnityEngine.Serialization;

public class Sam : MonoBehaviour
{
    [SerializeField] public Texture2D _inputTexture;
    [SerializeField] public GameObject _inputObject;
    [SerializeField] public GameObject _outputObject;
    [SerializeField] public Vector2 SegmentLocation;
    [SerializeField] public ModelAsset _samEncoderAsset;
    [SerializeField] public ModelAsset _samDecoderAsset;
}
