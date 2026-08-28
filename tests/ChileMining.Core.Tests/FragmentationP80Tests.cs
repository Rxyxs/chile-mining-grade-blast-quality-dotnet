using System;
using System.IO;
using System.Linq;
using ChileMining.Core.Data;
using ChileMining.Core.Generation;
using ChileMining.Core.Ml;
using Xunit;

namespace ChileMining.Core.Tests;

public class FragmentationP80Tests
{
    [Fact]
    public void HigherPowderFactor_ProducesLowerP80_OnAverage()
    {
        // Reemplaza a la vieja asercion sobre CalidadFragmentacion=="Fino":
        // chequea directamente la magnitud fisica (P80 continuo), no un
        // bucket categorico cuyos umbrales pueden recalibrarse.
        var designs = SyntheticDataGenerator.GenerateBlastDesigns(count: 3000, seed: 43);
        var ordenado = designs.OrderBy(d => d.FactorPotenciaKgTon).ToList();

        double p80PromedioBajo = ordenado.Take(ordenado.Count / 2).Average(d => d.P80Cm);
        double p80PromedioAlto = ordenado.Skip(ordenado.Count / 2).Average(d => d.P80Cm);

        Assert.True(
            p80PromedioAlto < p80PromedioBajo,
            $"Mayor factor de potencia deberia producir menor P80 promedio ({p80PromedioAlto:F1} vs {p80PromedioBajo:F1})");
    }

    [Theory]
    [InlineData(29.9f, "Fino")]
    [InlineData(30f, "Fino")]
    [InlineData(30.1f, "Medio")]
    [InlineData(45f, "Medio")]
    [InlineData(45.1f, "Grueso")]
    [InlineData(60f, "Grueso")]
    [InlineData(60.1f, "SobreTamano")]
    public void FragmentationQuality_Classify_UsesExpectedBoundaries(float p80, string expected)
    {
        Assert.Equal(expected, FragmentationQuality.Classify(p80));
    }

    [Fact]
    public void P80Estimator_TrainsWithReasonableFit()
    {
        var designs = SyntheticDataGenerator.GenerateBlastDesigns(count: 2000, seed: 43);
        var estimator = new FragmentationP80Estimator();
        var metrics = estimator.TrainAndEvaluate(designs);

        // El generador es determinista (formula Kuz-Ram + 5% de ruido gaussiano),
        // asi que un regresor razonable deberia explicar la mayor parte de la
        // varianza -- un R2 bajo indicaria un bug real en el pipeline o las features.
        Assert.True(metrics.RSquared > 0.85, $"R2 inesperadamente bajo: {metrics.RSquared:F4}");
    }

    [Fact]
    public void OnnxExport_ProducesPredictionsMatchingMLNetWithinTolerance()
    {
        var designs = SyntheticDataGenerator.GenerateBlastDesigns(count: 1500, seed: 43);
        var estimator = new FragmentationP80Estimator();
        estimator.TrainAndEvaluate(designs);

        string onnxPath = Path.Combine(Path.GetTempPath(), $"p80_test_{Guid.NewGuid():N}.onnx");
        try
        {
            estimator.ExportToOnnx(onnxPath, designs.Take(10));
            Assert.True(File.Exists(onnxPath));

            using var onnxService = new OnnxP80InferenceService(onnxPath);

            foreach (var design in designs.Skip(1400).Take(15))
            {
                float mlNetPrediction = estimator.Predict(design).P80CmEstimado;
                float onnxPrediction = onnxService.PredictP80Cm(design);

                // Diferencia de precision flotante entre el grafo ML.NET nativo
                // y su traduccion a ONNX, no una discrepancia de logica -- si
                // el export/import de features estuviera roto, la diferencia
                // seria de ordenes de magnitud, no de centesimas.
                Assert.True(
                    Math.Abs(mlNetPrediction - onnxPrediction) < 0.01f,
                    $"ML.NET={mlNetPrediction:F4} vs ONNX={onnxPrediction:F4}, diferencia excede tolerancia");
            }
        }
        finally
        {
            if (File.Exists(onnxPath)) File.Delete(onnxPath);
        }
    }
}
