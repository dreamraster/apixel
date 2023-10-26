using aPiXeL;
using Unity.Sentis;
using UnityEngine;

public class aPiXeLDiffusion : MonoBehaviour
{

    [SerializeField] public GameObject _target;
    [SerializeField] public ModelAsset _unetModel;
    [SerializeField] public ModelAsset _textEncoderModel;
    [SerializeField] public ModelAsset _vaeModel;
    [SerializeField] public ModelAsset _textTokenizerModel;
    [SerializeField] public string _prompt;
    [SerializeField] public int _seed;
    [SerializeField] public int _steps;

    public string Prompt() => _prompt;
    public int Seed() => _seed;
    public int Steps() => _steps;
    public GameObject Target() => _target;
}
