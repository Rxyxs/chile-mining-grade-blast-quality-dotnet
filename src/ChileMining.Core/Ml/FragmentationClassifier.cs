using System;
using System.Collections.Generic;
using ChileMining.Core.Data;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ChileMining.Core.Ml;

/// <summary>
/// Clasificacion multiclase (SDCA) de la calidad de fragmentacion esperada de una
/// tronadura (Fino / Medio / Grueso / SobreTamano) a partir de sus parametros de diseno.
/// </summary>
public class FragmentationClassifier
{
    private readonly MLContext _mlContext;
    private ITransformer? _model;
    private DataViewSchema? _dataSchema;

    public FragmentationClassifier(int seed = 2)
    {
        _mlContext = new MLContext(seed: seed);
    }

    public MulticlassClassificationMetrics TrainAndEvaluate(IEnumerable<BlastDesign> designs, double testFraction = 0.2)
    {
        var data = _mlContext.Data.LoadFromEnumerable(designs);
        _dataSchema = data.Schema;
        var split = _mlContext.Data.TrainTestSplit(data, testFraction: testFraction, seed: 7);

        var pipeline = BuildPipeline();
        _model = pipeline.Fit(split.TrainSet);

        var predictions = _model.Transform(split.TestSet);
        return _mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "LabelKey");
    }

    private IEstimator<ITransformer> BuildPipeline()
    {
        return _mlContext.Transforms.Conversion.MapValueToKey("LabelKey", "FragmentationLabel")
            .Append(_mlContext.Transforms.Concatenate(
                "Features",
                nameof(BlastDesign.BurdenM), nameof(BlastDesign.EspaciamientoM),
                nameof(BlastDesign.FactorPotenciaKgTon), nameof(BlastDesign.DurezaRocaMpa),
                nameof(BlastDesign.DiametroPerforacionMm)))
            .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "Features"))
            .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
    }

    public FragmentationPrediction Predict(BlastDesign design)
    {
        if (_model is null)
        {
            throw new InvalidOperationException("El modelo no ha sido entrenado ni cargado.");
        }

        var predictionEngine = _mlContext.Model.CreatePredictionEngine<BlastDesign, FragmentationPrediction>(_model);
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
        _model = _mlContext.Model.Load(path, out _);
    }
}
