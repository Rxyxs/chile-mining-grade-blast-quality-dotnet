using System;
using System.Collections.Generic;
using System.Linq;
using ChileMining.Core.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ChileMining.Core.Ml;

/// <summary>
/// Sirve inferencia de P80 directamente con Microsoft.ML.OnnxRuntime
/// (InferenceSession), sin pasar por MLContext -- el mismo camino de
/// inferencia que usaria un servicio productivo desplegado sin el runtime
/// completo de ML.NET, o un consumidor en otro lenguaje/plataforma que
/// hable ONNX (Python, C++, Java...). El modelo se exporta una vez desde
/// FragmentationP80Estimator.ExportToOnnx y se sirve aqui.
///
/// Nombres de tensores confirmados inspeccionando session.InputMetadata /
/// OutputMetadata del .onnx exportado (no asumidos): ML.NET nombra cada
/// input de tensor igual a la columna fuente, y el output de la cabeza de
/// regresion queda en "Score.output".
/// </summary>
public sealed class OnnxP80InferenceService : IDisposable
{
    private const string OutputTensorName = "Score.output";
    private readonly InferenceSession _session;

    public OnnxP80InferenceService(string onnxModelPath)
    {
        _session = new InferenceSession(onnxModelPath);
    }

    public float PredictP80Cm(BlastDesign design)
    {
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("BurdenM", new DenseTensor<float>(new[] { design.BurdenM }, new[] { 1, 1 })),
            NamedOnnxValue.CreateFromTensor("EspaciamientoM", new DenseTensor<float>(new[] { design.EspaciamientoM }, new[] { 1, 1 })),
            NamedOnnxValue.CreateFromTensor("FactorPotenciaKgTon", new DenseTensor<float>(new[] { design.FactorPotenciaKgTon }, new[] { 1, 1 })),
            NamedOnnxValue.CreateFromTensor("DurezaRocaMpa", new DenseTensor<float>(new[] { design.DurezaRocaMpa }, new[] { 1, 1 })),
            NamedOnnxValue.CreateFromTensor("DiametroPerforacionMm", new DenseTensor<float>(new[] { design.DiametroPerforacionMm }, new[] { 1, 1 })),
            NamedOnnxValue.CreateFromTensor("AlturaBancoM", new DenseTensor<float>(new[] { design.AlturaBancoM }, new[] { 1, 1 })),
        };

        using var results = _session.Run(inputs, new[] { OutputTensorName });
        return results[0].AsTensor<float>().First();
    }

    public void Dispose() => _session.Dispose();
}
