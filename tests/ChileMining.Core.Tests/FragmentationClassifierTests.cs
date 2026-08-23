using ChileMining.Core.Data;
using ChileMining.Core.Generation;
using ChileMining.Core.Ml;
using Xunit;

namespace ChileMining.Core.Tests;

public class FragmentationClassifierTests
{
    [Fact]
    public void TrainAndEvaluate_LearnsRealSignal_AboveRandomChance()
    {
        var designs = SyntheticDataGenerator.GenerateBlastDesigns(count: 2000, seed: 43);
        var classifier = new FragmentationClassifier();

        var metrics = classifier.TrainAndEvaluate(designs);

        // Con 4 clases, el azar da ~0.25 de accuracy. Exigimos bastante mas para
        // confirmar que el modelo aprendio la relacion causal factor de potencia /
        // burden / espaciamiento / dureza -> calidad de fragmentacion.
        Assert.True(metrics.MicroAccuracy > 0.5, $"MicroAccuracy demasiado baja: {metrics.MicroAccuracy:F3}");
    }

    [Fact]
    public void Predict_ReturnsValidCategory_ForTypicalBlastDesign()
    {
        var designs = SyntheticDataGenerator.GenerateBlastDesigns(count: 2000, seed: 43);
        var classifier = new FragmentationClassifier();
        classifier.TrainAndEvaluate(designs);

        var diseno = new BlastDesign
        {
            BurdenM = 5f,
            EspaciamientoM = 6f,
            FactorPotenciaKgTon = 0.30f,
            DurezaRocaMpa = 120f,
            DiametroPerforacionMm = 250f,
        };

        var prediccion = classifier.Predict(diseno);
        var categoriasValidas = new[] { "Fino", "Medio", "Grueso", "SobreTamano" };

        Assert.Contains(prediccion.CalidadPredicha, categoriasValidas);
    }
}
