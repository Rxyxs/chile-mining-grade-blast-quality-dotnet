using System;
using System.Collections.Generic;
using ChileMining.Core.Data;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ChileMining.Core.Ml;

/// <summary>
/// Regresion (FastTree) que estima la ley de cobre (%) a partir de datos de sondaje.
/// </summary>
public class GradeEstimator
{
    private readonly MLContext _mlContext;
    private ITransformer? _model;
    private DataViewSchema? _dataSchema;

    public GradeEstimator(int seed = 1)
    {
        _mlContext = new MLContext(seed: seed);
    }

    public RegressionMetrics TrainAndEvaluate(IEnumerable<DrillHoleSample> samples, double testFraction = 0.2)
    {
        var data = _mlContext.Data.LoadFromEnumerable(samples);
        _dataSchema = data.Schema;
        var split = _mlContext.Data.TrainTestSplit(data, testFraction: testFraction, seed: 7);

        var pipeline = BuildPipeline();
        _model = pipeline.Fit(split.TrainSet);

        var predictions = _model.Transform(split.TestSet);
        return _mlContext.Regression.Evaluate(predictions, labelColumnName: "Label");
    }

    private IEstimator<ITransformer> BuildPipeline()
    {
        return _mlContext.Transforms.Categorical.OneHotEncoding("UnidadGeologicaEncoded", nameof(DrillHoleSample.UnidadGeologica))
            .Append(_mlContext.Transforms.Categorical.OneHotEncoding("TipoAlteracionEncoded", nameof(DrillHoleSample.TipoAlteracion)))
            .Append(_mlContext.Transforms.Concatenate(
                "Features",
                "UnidadGeologicaEncoded", "TipoAlteracionEncoded",
                nameof(DrillHoleSample.ProfundidadM), nameof(DrillHoleSample.DensidadGrCm3),
                nameof(DrillHoleSample.ResistividadOhmM), nameof(DrillHoleSample.DistanciaFallaM)))
            .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(_mlContext.Regression.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"));
    }

    public GradePrediction Predict(DrillHoleSample sample)
    {
        if (_model is null)
        {
            throw new InvalidOperationException("El modelo no ha sido entrenado ni cargado.");
        }

        var predictionEngine = _mlContext.Model.CreatePredictionEngine<DrillHoleSample, GradePrediction>(_model);
        return predictionEngine.Predict(sample);
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
