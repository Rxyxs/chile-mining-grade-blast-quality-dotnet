using ChileMining.Core.Data;
using ChileMining.Core.Generation;
using ChileMining.Core.Ml;
using System.Globalization;
using ScottPlot;

namespace ChileMining.Charts;

/// <summary>
/// Genera los graficos del README, en .NET, desde los mismos modelos que entrena
/// ChileMining.Trainer. Ejecutar con:
///   dotnet run --project src/ChileMining.Charts
///
/// Esta separado del Trainer a proposito: el Trainer hace una corrida unica y su
/// salida es la que el README cita, mientras que estos graficos necesitan repetir
/// el entrenamiento varias veces para medir la variacion entre corridas. Mantener
/// la dependencia de ScottPlot fuera del Trainer tambien evita que el pipeline de
/// entrenamiento arrastre una libreria de graficos.
/// </summary>
public static class Program
{
    private const int Runs = 12;
    private const int DrillSeed = 42;
    private const int BlastSeed = 43;
    private const int HoldoutDrillSeed = 942;
    private const int HoldoutBlastSeed = 943;

    private static readonly Color Ink = Color.FromHex("#2B2B2B");
    private static readonly Color Grid = Color.FromHex("#D9D9D9");
    private static readonly Color FastTreeColor = Color.FromHex("#4C7A3E");
    private static readonly Color SdcaColor = Color.FromHex("#B5553D");
    private static readonly Color OgdColor = Color.FromHex("#8FA8B8");
    private static readonly Color Accent = Color.FromHex("#B58900");

    public static int Main()
    {
        // README.md (el ingles) es el que GitHub muestra por defecto y el que incrusta
        // estos graficos, asi que los rotulos y el separador decimal van en cultura
        // invariante, no en la del equipo que los genera.
        CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
        Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;

        var root = FindRepositoryRoot();
        var figDir = Path.Combine(root, "outputs", "figures");
        Directory.CreateDirectory(figDir);
        Console.WriteLine($"Writing figures to: {figDir}\n");

        var drillHoles = SyntheticDataGenerator.GenerateDrillHoles(count: 2000, seed: DrillSeed);
        var blastDesigns = SyntheticDataGenerator.GenerateBlastDesigns(count: 2000, seed: BlastSeed);

        TrainerStability(drillHoles, figDir);
        ClassifierStability(blastDesigns, figDir);
        P80PredictedVsActual(blastDesigns, figDir);
        GradePredictedVsActual(drillHoles, figDir);

        Console.WriteLine("\nListo.");
        return 0;
    }

    // ------------------------------------------------------------------
    // 1. Cual trainer gana, y cuanto de esa respuesta es ruido
    // ------------------------------------------------------------------
    private static void TrainerStability(IReadOnlyList<DrillHoleSample> drillHoles, string figDir)
    {
        Console.WriteLine($"1/4 trainer_comparison_stability ({Runs} corridas)...");

        var byTrainer = new Dictionary<string, List<double>>();
        for (var run = 0; run < Runs; run++)
        {
            foreach (var result in GradeEstimatorTrainerComparison.Compare(drillHoles))
            {
                if (!byTrainer.TryGetValue(result.TrainerName, out var list))
                {
                    list = new List<double>();
                    byTrainer[result.TrainerName] = list;
                }
                list.Add(result.Metrics.RSquared);
            }
        }

        var names = byTrainer.Keys.OrderByDescending(n => byTrainer[n].Average()).ToList();
        var plot = NewPlot();

        for (var i = 0; i < names.Count; i++)
        {
            var values = byTrainer[names[i]];
            var color = names[i].StartsWith("FastTree", StringComparison.Ordinal) ? FastTreeColor
                      : names[i].StartsWith("SDCA", StringComparison.Ordinal) ? SdcaColor
                      : OgdColor;

            var xs = values.Select(_ => (double)i).ToArray();
            var scatter = plot.Add.ScatterPoints(Jitter(xs, i), values.ToArray());
            scatter.Color = color.WithAlpha(0.55);
            scatter.MarkerSize = 9;

            var min = values.Min();
            var max = values.Max();
            var mean = values.Average();

            var bar = plot.Add.Line(i - 0.22, mean, i + 0.22, mean);
            bar.Color = color;
            bar.LineWidth = 3;

            var label = plot.Add.Text(
                max - min < 1e-9
                    ? $"{mean:F4}\n(identical across all {Runs})"
                    : $"{mean:F4}\n[{min:F4}, {max:F4}]",
                i, max + 0.0045);
            label.LabelFontSize = 11;
            label.LabelFontColor = color;
            label.LabelBold = true;
            label.Alignment = Alignment.LowerCenter;
        }

        plot.Axes.Bottom.SetTicks(
            Enumerable.Range(0, names.Count).Select(i => (double)i).ToArray(),
            names.ToArray());
        plot.YLabel("GradeEstimator R²");
        plot.Title($"SDCA beats FastTree, and the margin survives the noise\n{Runs} retrainings over the same 2,000 drill-hole samples");
        plot.Axes.SetLimitsY(
            byTrainer.Values.SelectMany(v => v).Min() - 0.006,
            byTrainer.Values.SelectMany(v => v).Max() + 0.014);

        Save(plot, Path.Combine(figDir, "trainer_comparison_stability.png"), 1000, 620);

        foreach (var n in names)
        {
            var v = byTrainer[n];
            Console.WriteLine($"    {n,-22} media={v.Average():F4}  min={v.Min():F4}  max={v.Max():F4}  rango={v.Max() - v.Min():F4}");
        }
    }

    // ------------------------------------------------------------------
    // 2. Las metricas del clasificador no son un numero fijo
    // ------------------------------------------------------------------
    private static void ClassifierStability(IReadOnlyList<BlastDesign> designs, string figDir)
    {
        Console.WriteLine($"2/4 classifier_metric_stability ({Runs} corridas)...");

        var micro = new List<double>();
        var macro = new List<double>();
        var logLoss = new List<double>();
        for (var run = 0; run < Runs; run++)
        {
            var metrics = new FragmentationClassifier().TrainAndEvaluate(designs);
            micro.Add(metrics.MicroAccuracy);
            macro.Add(metrics.MacroAccuracy);
            logLoss.Add(metrics.LogLoss);
        }

        var plot = NewPlot();
        var series = new (string Name, List<double> Values, Color Color)[]
        {
            ("MicroAccuracy", micro, FastTreeColor),
            ("MacroAccuracy", macro, Accent),
            ("LogLoss", logLoss, SdcaColor),
        };

        for (var i = 0; i < series.Length; i++)
        {
            var (name, values, color) = series[i];
            var xs = values.Select(_ => (double)i).ToArray();
            var scatter = plot.Add.ScatterPoints(Jitter(xs, i), values.ToArray());
            scatter.Color = color.WithAlpha(0.55);
            scatter.MarkerSize = 9;

            var mean = values.Average();
            var bar = plot.Add.Line(i - 0.22, mean, i + 0.22, mean);
            bar.Color = color;
            bar.LineWidth = 3;

            var text = plot.Add.Text(
                $"{mean:F4}\n[{values.Min():F4}, {values.Max():F4}]\nspread {values.Max() - values.Min():F4}",
                i, values.Max() + 0.012);
            text.LabelFontSize = 10.5f;
            text.LabelFontColor = color;
            text.LabelBold = true;
            text.Alignment = Alignment.LowerCenter;
        }

        plot.Axes.Bottom.SetTicks(
            Enumerable.Range(0, series.Length).Select(i => (double)i).ToArray(),
            series.Select(s => s.Name).ToArray());
        plot.YLabel("Metric value");
        plot.Title($"The multiclass SDCA classifier does not return a fixed number\n{Runs} retrainings over the same 2,000 blast designs");
        var all = series.SelectMany(s => s.Values).ToList();
        plot.Axes.SetLimitsY(all.Min() - 0.03, all.Max() + 0.075);

        Save(plot, Path.Combine(figDir, "classifier_metric_stability.png"), 1000, 620);

        foreach (var (name, values, _) in series)
        {
            Console.WriteLine($"    {name,-15} media={values.Average():F4}  min={values.Min():F4}  max={values.Max():F4}  rango={values.Max() - values.Min():F4}");
        }
    }

    // ------------------------------------------------------------------
    // 3. P80: lo predicho contra lo real, fuera de muestra
    // ------------------------------------------------------------------
    private static void P80PredictedVsActual(IReadOnlyList<BlastDesign> designs, string figDir)
    {
        Console.WriteLine("3/4 p80_predicted_vs_actual...");

        var estimator = new FragmentationP80Estimator();
        var metrics = estimator.TrainAndEvaluate(designs);

        // Holdout generado con otra semilla: ninguna de estas filas participo del
        // entrenamiento ni del split interno, asi que la nube es honestamente
        // fuera de muestra.
        var holdout = SyntheticDataGenerator.GenerateBlastDesigns(count: 600, seed: HoldoutBlastSeed);
        var actual = holdout.Select(d => (double)d.P80Cm).ToArray();
        var predicted = holdout.Select(d => (double)estimator.Predict(d).P80CmEstimado).ToArray();

        var plot = ScatterWithIdentity(
            actual, predicted, Accent,
            "Actual P80 (cm)", "Predicted P80 (cm)",
            $"FragmentationP80Estimator on {holdout.Count} out-of-sample blast designs\n" +
            $"internal R² {metrics.RSquared:F4} · RMSE {metrics.RootMeanSquaredError:F2} cm · " +
            $"holdout R² {RSquared(actual, predicted):F4}");

        Save(plot, Path.Combine(figDir, "p80_predicted_vs_actual.png"), 760, 720);
        Console.WriteLine($"    R2 interno={metrics.RSquared:F4}  RMSE={metrics.RootMeanSquaredError:F4}  R2 holdout={RSquared(actual, predicted):F4}");
    }

    // ------------------------------------------------------------------
    // 4. Ley de Cu: lo predicho contra lo real, fuera de muestra
    // ------------------------------------------------------------------
    private static void GradePredictedVsActual(IReadOnlyList<DrillHoleSample> drillHoles, string figDir)
    {
        Console.WriteLine("4/4 grade_predicted_vs_actual...");

        var estimator = new GradeEstimator();
        var metrics = estimator.TrainAndEvaluate(drillHoles);

        var holdout = SyntheticDataGenerator.GenerateDrillHoles(count: 600, seed: HoldoutDrillSeed);
        var actual = holdout.Select(s => (double)s.LeyCuPct).ToArray();
        var predicted = holdout.Select(s => (double)estimator.Predict(s).LeyCuPctEstimada).ToArray();

        var plot = ScatterWithIdentity(
            actual, predicted, FastTreeColor,
            "Actual Cu grade (%)", "Predicted Cu grade (%)",
            $"GradeEstimator on {holdout.Count} out-of-sample drill-hole samples\n" +
            $"internal R² {metrics.RSquared:F4} · RMSE {metrics.RootMeanSquaredError:F3} pp · " +
            $"holdout R² {RSquared(actual, predicted):F4}");

        Save(plot, Path.Combine(figDir, "grade_predicted_vs_actual.png"), 760, 720);
        Console.WriteLine($"    R2 interno={metrics.RSquared:F4}  RMSE={metrics.RootMeanSquaredError:F4}  R2 holdout={RSquared(actual, predicted):F4}");
    }

    // ------------------------------------------------------------------
    // Utilidades
    // ------------------------------------------------------------------
    private static Plot NewPlot()
    {
        var plot = new Plot();
        plot.FigureBackground.Color = Colors.White;
        plot.DataBackground.Color = Colors.White;
        plot.Axes.Color(Ink);
        plot.Grid.MajorLineColor = Grid.WithAlpha(0.7);
        return plot;
    }

    private static Plot ScatterWithIdentity(
        double[] actual, double[] predicted, Color color,
        string xLabel, string yLabel, string title)
    {
        var plot = NewPlot();

        var scatter = plot.Add.ScatterPoints(actual, predicted);
        scatter.Color = color.WithAlpha(0.4);
        scatter.MarkerSize = 6;

        var lo = Math.Min(actual.Min(), predicted.Min());
        var hi = Math.Max(actual.Max(), predicted.Max());
        var identity = plot.Add.Line(lo, lo, hi, hi);
        identity.Color = Ink.WithAlpha(0.65);
        identity.LineWidth = 2;
        identity.LinePattern = LinePattern.Dashed;

        var note = plot.Add.Text("perfect prediction", hi, hi);
        note.LabelFontSize = 10;
        note.LabelFontColor = Ink;
        note.Alignment = Alignment.LowerRight;

        plot.XLabel(xLabel);
        plot.YLabel(yLabel);
        plot.Title(title);
        plot.Axes.SetLimits(lo - (hi - lo) * 0.04, hi + (hi - lo) * 0.04,
                            lo - (hi - lo) * 0.04, hi + (hi - lo) * 0.08);
        return plot;
    }

    /// <summary>Separa horizontalmente los puntos que caen en el mismo x, para que
    /// corridas con el mismo valor no se tapen entre si.</summary>
    private static double[] Jitter(double[] xs, int seed)
    {
        var rng = new Random(seed);
        return xs.Select(x => x + (rng.NextDouble() - 0.5) * 0.3).ToArray();
    }

    private static double RSquared(double[] actual, double[] predicted)
    {
        var mean = actual.Average();
        var ssRes = actual.Zip(predicted, (a, p) => (a - p) * (a - p)).Sum();
        var ssTot = actual.Sum(a => (a - mean) * (a - mean));
        return 1 - ssRes / ssTot;
    }

    private static void Save(Plot plot, string path, int width, int height)
    {
        plot.SavePng(path, width, height);
        Console.WriteLine($"  escrito outputs/figures/{Path.GetFileName(path)}");
    }

    private static string FindRepositoryRoot()
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
