using System;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

using Unity.Barracuda;
using UnityEngine.Profiling;
using System.Collections;
using System.Collections.Generic;

/// <summary>
/// This component tests getting the latest camera image
/// and converting it to RGBA format. If successful,
/// it displays the image on the screen as a RawImage
/// and also displays information about the image.
///
/// This is useful for computer vision applications where
/// you need to access the raw pixels from camera image
/// on the CPU.
///
/// This is different from the ARCameraBackground component, which
/// efficiently displays the camera image on the screen. If you
/// just want to blit the camera texture to the screen, use
/// the ARCameraBackground, or use Graphics.Blit to create
/// a GPU-friendly RenderTexture.
///
/// In this example, we get the camera image data on the CPU,
/// convert it to an RGBA format, then display it on the screen
/// as a RawImage texture to demonstrate it is working.
/// This is done as an example; do not use this technique simply
/// to render the camera image on screen.
/// </summary>
public class TestCameraImage : MonoBehaviour
{
    [SerializeField]
    [Tooltip("The ARCameraManager which will produce frame events.")]
    ARCameraManager m_CameraManager;

    // Baracuda
    public float epsilon = 1e-3f;
    public NNModel srcModel;
    //spublic Texture2D inputImage;
    public TextAsset labelsAsset;
    public int inputResolutionY = 224;
    public int inputResolutionX = 224;
    public Material preprocessMaterial;

    public bool useGPU = true;

    private string imgInfo;
    //public RawImage displayImage;
    private Texture2D m_barracudaTexture;
    public Button m_YourFirstButton;

    //public int repeatExecution = 1000000;

    private Model model;
    private IWorker engine;
    private Dictionary<string, Tensor> inputs = new Dictionary<string, Tensor>();
    private string[] labels;

    private float averageDt;
    private float rawAverageDt;
    private bool flagFrameRdy;

    /// <summary>
    /// Get or set the <c>ARCameraManager</c>.
    /// </summary>
    public ARCameraManager cameraManager
    {
        get { return m_CameraManager; }
        set { m_CameraManager = value; }
    }

    [SerializeField]
    RawImage m_RawImage;

    /// <summary>
    /// The UI RawImage used to display the image on screen.
    /// </summary>
    public RawImage rawImage
    {
        get { return m_RawImage; }
        set { m_RawImage = value; }
    }

    [SerializeField]
    Text m_ImageInfo;

    /// <summary>
    /// The UI Text used to display information about the image on screen.
    /// </summary>
    public Text imageInfo
    {
        get { return m_ImageInfo; }
        set { m_ImageInfo = value; }
    }

    void OnEnable()
    {
        if (m_CameraManager != null)
        {
            m_CameraManager.frameReceived += OnCameraFrameReceived;
        }
    }

    void OnDisable()
    {
        if (m_CameraManager != null)
        {
            m_CameraManager.frameReceived -= OnCameraFrameReceived;
        }
    }
    IEnumerator Start()
    {
        //workaround button
        m_YourFirstButton.onClick.AddListener(TaskOnClick);

        Application.targetFrameRate = 60;

        labels = labelsAsset.text.Split('\n');
        model = ModelLoader.Load(srcModel, false);
        engine = WorkerFactory.CreateWorker(model, useGPU ? WorkerFactory.Device.GPU : WorkerFactory.Device.CPU);

        //var input = new Tensor(PrepareTextureForInput(m_Texture, !useGPU), 3);

        //inputs["input"] = input;

        yield return null;

        flagFrameRdy = true;
        StartCoroutine(RunInference());
    }
    void TaskOnClick()
    {
        //Output this to console when Button1 or Button3 is clicked
        Debug.Log("You have clicked the button!");
        flagFrameRdy = true;
    }
    unsafe void OnCameraFrameReceived(ARCameraFrameEventArgs eventArgs)
    {
        // Attempt to get the latest camera image. If this method succeeds,
        // it acquires a native resource that must be disposed (see below).
        XRCameraImage image;
        if (!cameraManager.TryGetLatestImage(out image))
        {
            return;
        }
  
        // Display some information about the camera image
        imgInfo = string.Format(
            "Image info:\n\twidth: {0}\n\theight: {1}\n\tplaneCount: {2}\n\ttimestamp: {3}\n\tformat: {4}",
            image.width, image.height, image.planeCount, image.timestamp, image.format);

        // Once we have a valid XRCameraImage, we can access the individual image "planes"
        // (the separate channels in the image). XRCameraImage.GetPlane provides
        // low-overhead access to this data. This could then be passed to a
        // computer vision algorithm. Here, we will convert the camera image
        // to an RGBA texture and draw it on the screen.

        // Choose an RGBA format.
        // See XRCameraImage.FormatSupported for a complete list of supported formats.
        var format = TextureFormat.RGBA32;

        if (m_Texture == null || m_Texture.width != image.width || m_Texture.height != image.height)
        {
            m_Texture = new Texture2D(image.width, image.height, format, false);
        }

        // Convert the image to format, flipping the image across the Y axis.
        // We can also get a sub rectangle, but we'll get the full image here.
        var conversionParams = new XRCameraImageConversionParams(image, format, CameraImageTransformation.MirrorY);

        // Texture2D allows us write directly to the raw texture data
        // This allows us to do the conversion in-place without making any copies.
        var rawTextureData = m_Texture.GetRawTextureData<byte>();
        try
        {
            image.Convert(conversionParams, new IntPtr(rawTextureData.GetUnsafePtr()), rawTextureData.Length);


        }
        finally
        {
            // We must dispose of the XRCameraImage after we're finished
            // with it to avoid leaking native resources.
            image.Dispose();
        }

        // Apply the updated texture data to our texture
        m_Texture.Apply();

        // Set the RawImage's texture so we can visualize it.
        m_RawImage.texture = m_Texture;
        m_barracudaTexture = m_Texture; //copy to prevent simultanious access of texture (but apparently not the reason for the app crashing)
    }

    Texture PrepareTextureForInput(Texture2D src, bool needsCPUcopy)
    {
        var targetRT = RenderTexture.GetTemporary(inputResolutionX, inputResolutionY, 0, RenderTextureFormat.ARGB32);
        RenderTexture.active = targetRT;
        Graphics.Blit(src, targetRT, preprocessMaterial);

        if (!needsCPUcopy)
            return targetRT;

        var result = new Texture2D(targetRT.width, targetRT.height);
        result.ReadPixels(new Rect(0, 0, targetRT.width, targetRT.height), 0, 0);
        result.Apply();

        return result;
    }
    IEnumerator RunInference()
    {
        // Skip frame before starting
        //yield return null;
        //m_RawImage.texture = m_Texture;

        while (true)//repeatExecution-- > 0)
        {
            if (flagFrameRdy)
            {
                var input = new Tensor(PrepareTextureForInput(m_barracudaTexture, !useGPU), 3);

                inputs["input"] = input;

                //yield return null;

                try
                {
                    var start = Time.realtimeSinceStartup;

                    Profiler.BeginSample("Schedule execution");
                    engine.Execute(inputs);
                    Profiler.EndSample();

                    Profiler.BeginSample("Fetch execution results");
                    var output = engine.PeekOutput();
                    Profiler.EndSample();

                    var res = output.ArgMax()[0];
                    var end = Time.realtimeSinceStartup;
                    var label = labels[res];

                    if (label.Contains("hotdog") && Mathf.Abs(output[res] - 0.578f) < epsilon)
                    {
                        m_ImageInfo.color = Color.green;
                        m_ImageInfo.text = $"Success: {labels[res]} {output[res] * 100}% \n" + imgInfo;
                    }
                    else
                    {
                        m_ImageInfo.color = Color.red;
                        m_ImageInfo.text = $"Failed: {labels[res]} {output[res] * 100}% \n" + imgInfo;
                    }

                    UpdateAverage(end - start);
                    Debug.Log($"frametime = {(end - start) * 1000f}ms, average = {averageDt * 1000}ms");


                }
                catch (Exception e)
                {
                    Debug.Log($"Exception happened {e}");
                    //throw;
                }
                flagFrameRdy = false;

            }

            yield return null;
        }
    }
    private void UpdateAverage(float newValue)
    {
        rawAverageDt = rawAverageDt * 0.9f + 0.1f * newValue;

        // Drop spikes above 20%
        if (newValue < 1.2f * rawAverageDt)
        {
            averageDt = averageDt * 0.9f + 0.1f * newValue;
        }
    }
    private void OnDestroy()
    {
        engine?.Dispose();

        foreach (var key in inputs.Keys)
        {
            inputs[key].Dispose();
        }

        inputs.Clear();
    }
    Texture2D m_Texture;
}
