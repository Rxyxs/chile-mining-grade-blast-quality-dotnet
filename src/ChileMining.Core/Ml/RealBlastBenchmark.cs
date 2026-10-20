using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ChileMining.Core.Ml;

/// <summary>
/// Una tronadura real publicada: los siete parametros adimensionales de diseno
/// y roca, y el tamano medio de fragmento medido (x50).
/// </summary>
public class RealBlast
{
    public string BlastId { get; set; } = string.Empty;
    public string Site { get; set; } = string.Empty;
    public float SOverB { get; set; }
    public float HOverB { get; set; }
    public float BOverD { get; set; }
    public float TOverB { get; set; }
    public float PowderFactorKgM3 { get; set; }
    public float InSituBlockM { get; set; }
    public float YoungModulusGpa { get; set; }

    [ColumnName("Label")]
    public float X50M { get; set; }

    /// <summary>x50 que predijo el modelo de Kuznetsov en el paper original (solo en el hold-out).</summary>
    [NoColumn]
    public float? PublishedKuzRamM { get; set; }

    /// <summary>x50 que predijo la regresion multivariada del paper original (solo en el hold-out).</summary>
    [NoColumn]
    public float? PublishedRegressionM { get; set; }
}

public class X50Prediction
{
    [ColumnName("Score")]
    public float X50M { get; set; }
}

public record BenchmarkRow(string Protocol, string Model, int N, double VarianceExplained, double MaeM, double MapePct);

/// <summary>
/// Compara el modelo de fragmentacion con tronaduras reales en vez del generador
/// sintetico. Datos: Hudaverdi, Kulatilake y Kuzu (2010), Int. J. Numer. Anal. Meth.
/// Geomech. 35:1318-1333, doi:10.1002/nag.957 -- 97 tronaduras de diez faenas en
/// cuatro continentes -- mas su hold-out publicado y cinco tronaduras de produccion
/// de Sui et al. (2025), Appl. Sci. 15:1254, CC BY 4.0. Transcripcion verificada en
/// github.com/fsantibanezleal/CAOS_BlastFrag (MIT).
///
/// Lo que importa es el protocolo. Un split aleatorio mezcla tronaduras de la misma
/// faena entre entrenamiento y prueba (hay filas casi identicas dentro de una faena),
/// asi que tambien se mide dejando una faena completa fuera cada vez: es la pregunta
/// real de "uso el modelo en una mina nueva".
///
/// La metrica es varianza explicada, 1 - SSE/SST respecto de la media de la prueba,
/// no el cuadrado de una correlacion: puede ser negativa si el modelo es peor que
/// predecir el promedio.
/// </summary>
public static class RealBlastBenchmark
{
    public static List<RealBlast> LoadCsv(string path)
    {
        var lines = File.ReadAllLines(path).Where(l => !l.StartsWith('#') && l.Trim().Length > 0).ToList();
        var header = lines[0].Split(',');
        int Col(string name) => Array.IndexOf(header, name);
        float F(string[] cells, string name) => float.Parse(cells[Col(name)], CultureInfo.InvariantCulture);
        float? Opt(string[] cells, string name) =>
            Col(name) >= 0 && cells[Col(name)].Length > 0 ? F(cells, name) : null;

        string labelColumn = Col("x50_m") >= 0 ? "x50_m" : "x50_measured_m";
        var blasts = new List<RealBlast>();
        foreach (var line in lines.Skip(1))
        {
            var c = line.Split(',');
            blasts.Add(new RealBlast
            {
                BlastId = c[Col("blast_id")],
                Site = c[Col("site")],
                SOverB = F(c, "S_over_B"),
                HOverB = F(c, "H_over_B"),
                BOverD = F(c, "B_over_D"),
                TOverB = F(c, "T_over_B"),
                PowderFactorKgM3 = F(c, "Pf_kg_m3"),
                InSituBlockM = F(c, "XB_m"),
                YoungModulusGpa = F(c, "E_GPa"),
                X50M = F(c, labelColumn),
                PublishedKuzRamM = Opt(c, "x50_kuzram_2010_m"),
                PublishedRegressionM = Opt(c, "x50_regression_2010_m"),
            });
        }

        return blasts;
    }

    public static double VarianceExplained(IReadOnlyList<double> actual, IReadOnlyList<double> predicted)
    {
        double mean = actual.Average();
        double sse = actual.Zip(predicted, (a, p) => (a - p) * (a - p)).Sum();
        double sst = actual.Sum(a => (a - mean) * (a - mean));
        return sst == 0 ? double.NaN : 1 - sse / sst;
    }

    private static BenchmarkRow Score(string protocol, string model, IReadOnlyList<double> actual, IReadOnlyList<double> predicted) =>
        new(protocol, model, actual.Count, VarianceExplained(actual, predicted),
            actual.Zip(predicted, (a, p) => Math.Abs(a - p)).Average(),
            actual.Zip(predicted, (a, p) => Math.Abs(a - p) / a).Average() * 100);

    private static IEstimator<ITransformer> Pipeline(MLContext ml, string trainer)
    {
        var features = ml.Transforms.Concatenate("Features",
                nameof(RealBlast.SOverB), nameof(RealBlast.HOverB), nameof(RealBlast.BOverD),
                nameof(RealBlast.TOverB), nameof(RealBlast.PowderFactorKgM3),
                nameof(RealBlast.InSituBlockM), nameof(RealBlast.YoungModulusGpa))
            .Append(ml.Transforms.NormalizeMinMax("Features"));
        return trainer switch
        {
            // Con 97 filas, hojas pequenas y pocos arboles: los valores por defecto de
            // FastTree (20 hojas, 10 ejemplos minimos) estan pensados para miles de filas.
            "FastTree" => features.Append(ml.Regression.Trainers.FastTree(
                numberOfLeaves: 6, numberOfTrees: 100, minimumExampleCountPerLeaf: 4, learningRate: 0.1)),
            "SDCA" => features.Append(ml.Regression.Trainers.Sdca()),
            _ => throw new ArgumentException(trainer),
        };
    }

    public static List<double> FitPredict(IReadOnlyList<RealBlast> train, IReadOnlyList<RealBlast> test, string trainer, int seed = 3)
    {
        var ml = new MLContext(seed: seed);
        var model = Pipeline(ml, trainer).Fit(ml.Data.LoadFromEnumerable(train));
        var scored = model.Transform(ml.Data.LoadFromEnumerable(test));
        return ml.Data.CreateEnumerable<X50Prediction>(scored, reuseRowObject: false).Select(p => (double)p.X50M).ToList();
    }

    public static List<BenchmarkRow> Run(IReadOnlyList<RealBlast> corpus, IReadOnlyList<RealBlast> holdout, IReadOnlyList<RealBlast> field)
    {
        var rows = new List<BenchmarkRow>();
        string[] trainers = { "FastTree", "SDCA" };

        // 1) 5-fold aleatorio (lo que se reporta habitualmente).
        var rng = new Random(42);
        var shuffled = corpus.OrderBy(_ => rng.Next()).ToList();
        foreach (var trainer in trainers.Append("Mean"))
        {
            var actual = new List<double>();
            var predicted = new List<double>();
            for (int k = 0; k < 5; k++)
            {
                var test = shuffled.Where((_, i) => i % 5 == k).ToList();
                var train = shuffled.Where((_, i) => i % 5 != k).ToList();
                actual.AddRange(test.Select(b => (double)b.X50M));
                predicted.AddRange(trainer == "Mean"
                    ? test.Select(_ => train.Average(b => (double)b.X50M))
                    : FitPredict(train, test, trainer));
            }

            rows.Add(Score("random 5-fold", trainer, actual, predicted));
        }

        // 2) Dejando una faena completa fuera (mina nueva).
        foreach (var trainer in trainers.Append("Mean"))
        {
            var actual = new List<double>();
            var predicted = new List<double>();
            foreach (var site in corpus.Select(b => b.Site).Distinct())
            {
                var test = corpus.Where(b => b.Site == site).ToList();
                var train = corpus.Where(b => b.Site != site).ToList();
                actual.AddRange(test.Select(b => (double)b.X50M));
                predicted.AddRange(trainer == "Mean"
                    ? test.Select(_ => train.Average(b => (double)b.X50M))
                    : FitPredict(train, test, trainer));
            }

            rows.Add(Score("leave one site out", trainer, actual, predicted));
        }

        // 3) Hold-out publicado: el modelo entrenado con las 97 contra las predicciones
        //    de Kuznetsov y de la regresion que publico el paper, en las mismas filas.
        var pub = holdout.Where(b => b.PublishedKuzRamM.HasValue).ToList();
        var pubActual = pub.Select(b => (double)b.X50M).ToList();
        foreach (var trainer in trainers)
        {
            rows.Add(Score("published hold-out", trainer, pubActual, FitPredict(corpus, pub, trainer)));
        }

        rows.Add(Score("published hold-out", "Kuz-Ram (as published)", pubActual, pub.Select(b => (double)b.PublishedKuzRamM!.Value).ToList()));
        rows.Add(Score("published hold-out", "Regression (as published)", pubActual, pub.Select(b => (double)b.PublishedRegressionM!.Value).ToList()));

        // 4) Cinco tronaduras de produccion de otra mina y otro continente.
        var fieldActual = field.Select(b => (double)b.X50M).ToList();
        foreach (var trainer in trainers)
        {
            rows.Add(Score("field blasts (Sui 2025)", trainer, fieldActual, FitPredict(corpus, field, trainer)));
        }

        return rows;
    }

    public static void SaveToCsv(IEnumerable<BenchmarkRow> rows, string path)
    {
        var lines = new List<string> { "protocol,model,n,variance_explained,mae_m,mape_pct" };
        lines.AddRange(rows.Select(r => string.Format(CultureInfo.InvariantCulture,
            "{0},{1},{2},{3:F3},{4:F3},{5:F1}", r.Protocol, r.Model, r.N, r.VarianceExplained, r.MaeM, r.MapePct)));
        File.WriteAllLines(path, lines);
    }
}
