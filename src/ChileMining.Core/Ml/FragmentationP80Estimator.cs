using System;
using System.Collections.Generic;
using System.IO;
using ChileMining.Core.Data;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ChileMining.Core.Ml;

/// <summary>
/// Regresion (FastTree) que estima P80 (cm) -- el tamano de malla de tamiz bajo
/// el cual pasa el 80% de la masa fragmentada, el KPI estandar de fragmentacion
/// de tronadura -- a partir de los parametros de diseno de la malla. A diferencia
/// de FragmentationClassifier (categoria discreta), este modelo predice el valor
/// continuo real que un ingeniero de tronadura reporta.
/// </summary>
public class FragmentationP80Estimator
{
    private readonly MLContext _mlContext;
    private ITransformer? _model;
    private DataViewSchema? _dataSchema;

    public FragmentationP80Estimator(int seed = 3)
    {
        _mlContext = new MLContext(seed: seed);
    }

    public RegressionMetrics TrainAndEvaluate(IEnumerable<BlastDesign> designs, double testFraction = 0.2)
    {
        var data = _mlContext.Data.LoadFromEnumerable(designs);
        _dataSchema = data.Schema;
        var split = _mlContext.Data.TrainTestSplit(data, testFraction: testFraction, seed: 7);

        var pipeline = BuildPipeline();
        _model = pipeline.Fit(split.TrainSet);

        var predictions = _model.Transform(split.TestSet);
        return _mlContext.Regression.Evaluate(predictions, labelColumnName: "Label");
    }

    private IEstimator<ITransformer> BuildPipeline()
    {
        return _mlContext.Transforms.Concatenate(
                "Features",
                nameof(BlastDesign.BurdenM), nameof(BlastDesign.EspaciamientoM),
                nameof(BlastDesign.FactorPotenciaKgTon), nameof(BlastDesign.DurezaRocaMpa),
                nameof(BlastDesign.DiametroPerforacionMm), nameof(BlastDesign.AlturaBancoM))
            .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(_mlContext.Regression.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"));
    }

    public P80Prediction Predict(BlastDesign design)
    {
        if (_model is null)
        {
            throw new InvalidOperationException("El modelo no ha sido entrenado ni cargado.");
        }

        var predictionEngine = _mlContext.Model.CreatePredictionEngine<BlastDesign, P80Prediction>(_model);
        return predictionEngine.Predict(design);
    }

    public void Save(string path)
    {
        if (_model is null || _dataSchema is null)
        {
            throw new InvalidOperationException("El modelo no ha sido entrenado.");
        }

        _mlContext.Model.Save(_model, _dataSchema, path);
    }

    public void Load(string path)
    {
        _model = _mlContext.Model.Load(path, out _dataSchema);
    }

    /// <summary>
    /// Exporta el pipeline entrenado a formato ONNX (.onnx), consumible por
    /// cualquier runtime compatible -- no solo ML.NET. `OnnxP80InferenceService`
    /// carga el archivo resultante con Microsoft.ML.OnnxRuntime directamente,
    /// sin pasar por MLContext, exactamente el flujo de un servicio de
    /// inferencia productivo que no necesita cargar el runtime completo de ML.NET.
    /// </summary>
    public void ExportToOnnx(string path, IEnumerable<BlastDesign> sampleData)
    {
        if (_model is null || _dataSchema is null)
        {
            throw new InvalidOperationException("El modelo no ha sido entrenado.");
        }

        // El sample view se recorta a solo las columnas de features numericas
        // (via SelectColumns) antes de pasarlo a ConvertToOnnx -- descubierto
        // corriendo la exportacion, no supuesto: pasar el BlastDesign completo
        // (que incluye la columna string FragmentationLabel, sin uso en el
        // pipeline de regresion) produce un grafo ONNX que revienta en tiempo
        // de inferencia con "OrtValue::Get IsTensorSequence() was false" en un
        // nodo Identity, porque el exportador intenta modelar el passthrough
        // de esa columna string igual. Recortar el schema de entrada al
        // exportar evita el nodo problematico por completo.
        var trimmed = _mlContext.Transforms
            .SelectColumns(
                nameof(BlastDesign.BurdenM), nameof(BlastDesign.EspaciamientoM),
                nameof(BlastDesign.FactorPotenciaKgTon), nameof(BlastDesign.DurezaRocaMpa),
                nameof(BlastDesign.DiametroPerforacionMm), nameof(BlastDesign.AlturaBancoM))
            .Fit(_mlContext.Data.LoadFromEnumerable(sampleData))
            .Transform(_mlContext.Data.LoadFromEnumerable(sampleData));

        using var stream = File.Create(path);
        _mlContext.Model.ConvertToOnnx(_model, trimmed, stream);
    }
}
