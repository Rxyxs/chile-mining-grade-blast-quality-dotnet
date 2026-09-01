using System;
using System.IO;
using System.Linq;
using ChileMining.Core.Generation;
using ChileMining.Core.Ml;

namespace ChileMining.Trainer;

/// <summary>
/// Orquestador: genera datos sinteticos -> entrena y evalua ambos modelos ML.NET ->
/// guarda datasets (.csv) y modelos entrenados (.zip) en data/.
/// Ejecutar desde cualquier lado con: dotnet run --project src/ChileMining.Trainer
/// </summary>
public static class Program
{
    public static void Main()
    {
        // En un contenedor no hay checkout de ChileMining.sln para que
        // FindRepoRoot() lo ubique -- CHILEMINING_DATA_DIR permite fijar el
        // directorio de salida explicitamente (ver Dockerfile). En desarrollo
        // local, sin la variable definida, se sigue usando data/ junto al
        // repo, como antes.
        string? overrideDir = Environment.GetEnvironmentVariable("CHILEMINING_DATA_DIR");
        string dataDir = overrideDir ?? Path.Combine(FindRepoRoot(), "data");
        Directory.CreateDirectory(dataDir);

        Console.WriteLine("=== 1/5 Generando datos sinteticos ===");
        var drillHoles = SyntheticDataGenerator.GenerateDrillHoles(count: 2000, seed: 42);
        var blastDesigns = SyntheticDataGenerator.GenerateBlastDesigns(count: 2000, seed: 43);
        SyntheticDataGenerator.SaveDrillHolesToCsv(drillHoles, Path.Combine(dataDir, "drill_holes.csv"));
        SyntheticDataGenerator.SaveBlastDesignsToCsv(blastDesigns, Path.Combine(dataDir, "blast_designs.csv"));
        Console.WriteLine($"  drill_holes.csv: {drillHoles.Count} filas");
        Console.WriteLine($"  blast_designs.csv: {blastDesigns.Count} filas");

        Console.WriteLine("\n=== 2/5 Entrenando GradeEstimator (regresion FastTree) ===");
        var gradeEstimator = new GradeEstimator();
        var regressionMetrics = gradeEstimator.TrainAndEvaluate(drillHoles);
        Console.WriteLine(FormattableString.Invariant($"  R-cuadrado: {regressionMetrics.RSquared:F4}"));
        Console.WriteLine(FormattableString.Invariant($"  RMSE: {regressionMetrics.RootMeanSquaredError:F4}"));
        Console.WriteLine(FormattableString.Invariant($"  MAE: {regressionMetrics.MeanAbsoluteError:F4}"));
        gradeEstimator.Save(Path.Combine(dataDir, "grade_estimator.zip"));

        Console.WriteLine("\n=== 2.1/5 Comparando trainers de regresion para GradeEstimator (FastTree vs SDCA vs OnlineGradientDescent) ===");
        var trainerComparison = GradeEstimatorTrainerComparison.Compare(drillHoles);
        foreach (var result in trainerComparison)
        {
            Console.WriteLine(FormattableString.Invariant(
                $"  {result.TrainerName,-22} R2={result.Metrics.RSquared:F4}  RMSE={result.Metrics.RootMeanSquaredError:F4}  MAE={result.Metrics.MeanAbsoluteError:F4}"));
        }
        GradeEstimatorTrainerComparison.SaveToCsv(trainerComparison, Path.Combine(dataDir, "grade_trainer_comparison.csv"));

        Console.WriteLine("\n=== 3/5 Entrenando FragmentationClassifier (SDCA multiclase) ===");
        var fragmentationClassifier = new FragmentationClassifier();
        var classificationMetrics = fragmentationClassifier.TrainAndEvaluate(blastDesigns);
        Console.WriteLine(FormattableString.Invariant($"  MicroAccuracy: {classificationMetrics.MicroAccuracy:F4}"));
        Console.WriteLine(FormattableString.Invariant($"  MacroAccuracy: {classificationMetrics.MacroAccuracy:F4}"));
        Console.WriteLine(FormattableString.Invariant($"  LogLoss: {classificationMetrics.LogLoss:F4}"));
        fragmentationClassifier.Save(Path.Combine(dataDir, "fragmentation_classifier.zip"));

        Console.WriteLine("\n=== 4/5 Entrenando FragmentationP80Estimator (regresion FastTree, P80 continuo) ===");
        var p80Estimator = new FragmentationP80Estimator();
        var p80Metrics = p80Estimator.TrainAndEvaluate(blastDesigns);
        Console.WriteLine(FormattableString.Invariant($"  R-cuadrado: {p80Metrics.RSquared:F4}"));
        Console.WriteLine(FormattableString.Invariant($"  RMSE: {p80Metrics.RootMeanSquaredError:F4} cm"));
        Console.WriteLine(FormattableString.Invariant($"  MAE: {p80Metrics.MeanAbsoluteError:F4} cm"));
        p80Estimator.Save(Path.Combine(dataDir, "p80_estimator.zip"));

        Console.WriteLine("\n=== 5/5 Exportando P80Estimator a ONNX ===");
        string onnxPath = Path.Combine(dataDir, "p80_estimator.onnx");
        p80Estimator.ExportToOnnx(onnxPath, blastDesigns.Take(10));
        Console.WriteLine($"  {onnxPath}");

        Console.WriteLine("\nListo. Modelos y datasets guardados en: " + dataDir);
    }

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "ChileMining.sln")))
        {
            dir = dir.Parent;
        }

        return dir?.FullName
            ?? throw new DirectoryNotFoundException("No se encontro ChileMining.sln en ningun directorio padre.");
    }
}
