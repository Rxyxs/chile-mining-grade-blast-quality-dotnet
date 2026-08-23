using ChileMining.Core.Generation;
using ChileMining.Core.Ml;
using Xunit;

namespace ChileMining.Core.Tests;

public class GradeEstimatorTests
{
    [Fact]
    public void TrainAndEvaluate_LearnsRealSignal_NotNearRandom()
    {
        var samples = SyntheticDataGenerator.GenerateDrillHoles(count: 2000, seed: 42);
        var estimator = new GradeEstimator();

        var metrics = estimator.TrainAndEvaluate(samples);

        // Un R2 cercano a 0 indicaria que el modelo no aprendio nada real de las
        // features (el mismo tipo de bug de "etiqueta no correlacionada con el texto"
        // encontrado antes en el clasificador de severidad del proyecto RAG).
        Assert.True(metrics.RSquared > 0.5, $"R-cuadrado demasiado bajo: {metrics.RSquared:F3}");
    }

    [Fact]
    public void Predict_ReturnsPositiveGrade_ForTypicalPorphyrySample()
    {
        var samples = SyntheticDataGenerator.GenerateDrillHoles(count: 2000, seed: 42);
        var estimator = new GradeEstimator();
        estimator.TrainAndEvaluate(samples);

        var muestra = new ChileMining.Core.Data.DrillHoleSample
        {
            ProfundidadM = 200f,
            UnidadGeologica = "Porfido Cu-Mo",
            TipoAlteracion = "Potasica",
            DensidadGrCm3 = 2.75f,
            ResistividadOhmM = 30f,
            DistanciaFallaM = 50f,
        };

        var prediccion = estimator.Predict(muestra);

        Assert.True(prediccion.LeyCuPctEstimada > 0f, "La ley estimada no deberia ser negativa ni cero.");
    }
}
