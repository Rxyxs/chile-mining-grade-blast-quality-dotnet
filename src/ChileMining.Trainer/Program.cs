using System;
using System.IO;
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
        string dataDir = Path.Combine(FindRepoRoot(), "data");
        Directory.CreateDirectory(dataDir);

        Console.WriteLine("=== 1/4 Generando datos sinteticos ===");
        var drillHoles = SyntheticDataGenerator.GenerateDrillHoles(count: 2000, seed: 42);
        var blastDesigns = SyntheticDataGenerator.GenerateBlastDesigns(count: 2000, seed: 43);
        SyntheticDataGenerator.SaveDrillHolesToCsv(drillHoles, Path.Combine(dataDir, "drill_holes.csv"));
        SyntheticDataGenerator.SaveBlastDesignsToCsv(blastDesigns, Path.Combine(dataDir, "blast_designs.csv"));
        Console.WriteLine($"  drill_holes.csv: {drillHoles.Count} filas");
        Console.WriteLine($"  blast_designs.csv: {blastDesigns.Count} filas");

        Console.WriteLine("\n=== 2/4 Entrenando GradeEstimator (regresion FastTree) ===");
        var gradeEstimator = new GradeEstimator();
        var regressionMetrics = gradeEstimator.TrainAndEvaluate(drillHoles);
        Console.WriteLine(FormattableString.Invariant($"  R-cuadrado: {regressionMetrics.RSquared:F4}"));
        Console.WriteLine(FormattableString.Invariant($"  RMSE: {regressionMetrics.RootMeanSquaredError:F4}"));
        Console.WriteLine(FormattableString.Invariant($"  MAE: {regressionMetrics.MeanAbsoluteError:F4}"));
        gradeEstimator.Save(Path.Combine(dataDir, "grade_estimator.zip"));

        Console.WriteLine("\n=== 3/4 Entrenando FragmentationClassifier (SDCA multiclase) ===");
        var fragmentationClassifier = new FragmentationClassifier();
        var classificationMetrics = fragmentationClassifier.TrainAndEvaluate(blastDesigns);
        Console.WriteLine(FormattableString.Invariant($"  MicroAccuracy: {classificationMetrics.MicroAccuracy:F4}"));
        Console.WriteLine(FormattableString.Invariant($"  MacroAccuracy: {classificationMetrics.MacroAccuracy:F4}"));
        Console.WriteLine(FormattableString.Invariant($"  LogLoss: {classificationMetrics.LogLoss:F4}"));
        fragmentationClassifier.Save(Path.Combine(dataDir, "fragmentation_classifier.zip"));

        Console.WriteLine("\n=== 4/4 Listo ===");
        Console.WriteLine($"Modelos y datasets guardados en: {dataDir}");
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
