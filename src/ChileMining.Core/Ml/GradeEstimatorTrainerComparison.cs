using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using ChileMining.Core.Data;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ChileMining.Core.Ml;

/// <summary>
/// Resultado de evaluar un trainer de regresion sobre el mismo split de datos.
/// </summary>
public record TrainerComparisonResult(string TrainerName, RegressionMetrics Metrics);

/// <summary>
/// Compara honestamente 3 trainers de regresion de ML.NET (FastTree, SDCA,
/// Online Gradient Descent) sobre el mismo pipeline de features y el mismo
/// train/test split que usa <see cref="GradeEstimator"/> -- para poder decidir
/// con datos, no por default, cual algoritmo usar para estimar ley de cobre.
/// No reemplaza a GradeEstimator (que sigue usando FastTree en produccion,
/// consumido por la app de escritorio); esto es un experimento aditivo que
/// corre aparte, vía ChileMining.Trainer.
/// </summary>
public static class GradeEstimatorTrainerComparison
{
    public static IReadOnlyList<TrainerComparisonResult> Compare(
        IEnumerable<DrillHoleSample> samples, double testFraction = 0.2, int seed = 1)
    {
        var mlContext = new MLContext(seed: seed);
        var data = mlContext.Data.LoadFromEnumerable(samples);
        var split = mlContext.Data.TrainTestSplit(data, testFraction: testFraction, seed: 7);

        var featurePipeline = mlContext.Transforms.Categorical.OneHotEncoding(
                "UnidadGeologicaEncoded", nameof(DrillHoleSample.UnidadGeologica))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(
                "TipoAlteracionEncoded", nameof(DrillHoleSample.TipoAlteracion)))
            .Append(mlContext.Transforms.Concatenate(
                "Features",
                "UnidadGeologicaEncoded", "TipoAlteracionEncoded",
                nameof(DrillHoleSample.ProfundidadM), nameof(DrillHoleSample.DensidadGrCm3),
                nameof(DrillHoleSample.ResistividadOhmM), nameof(DrillHoleSample.DistanciaFallaM)))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"));

        var results = new List<TrainerComparisonResult>();

        var trainers = new (string Name, IEstimator<ITransformer> Trainer)[]
        {
            ("FastTree", mlContext.Regression.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features")),
            ("SDCA", mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features")),
            ("OnlineGradientDescent", mlContext.Regression.Trainers.OnlineGradientDescent(labelColumnName: "Label", featureColumnName: "Features")),
        };

        foreach (var (name, trainer) in trainers)
        {
            var pipeline = featurePipeline.Append(trainer);
            var model = pipeline.Fit(split.TrainSet);
            var predictions = model.Transform(split.TestSet);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label");
            results.Add(new TrainerComparisonResult(name, metrics));
        }

        return results;
    }

    /// <summary>
    /// Guarda la comparacion como CSV en data/, consistente con el resto del
    /// proyecto (datasets y no hay capa de persistencia separada -- CSV es lo
    /// que ya se usa para todo lo generado por ChileMining.Trainer).
    /// </summary>
    public static void SaveToCsv(IReadOnlyList<TrainerComparisonResult> results, string path)
    {
        using var writer = new StreamWriter(path);
        writer.WriteLine("Trainer,RSquared,RMSE,MAE");
        foreach (var r in results)
        {
            writer.WriteLine(FormattableString.Invariant(
                $"{r.TrainerName},{r.Metrics.RSquared:F4},{r.Metrics.RootMeanSquaredError:F4},{r.Metrics.MeanAbsoluteError:F4}"));
        }
    }
}
