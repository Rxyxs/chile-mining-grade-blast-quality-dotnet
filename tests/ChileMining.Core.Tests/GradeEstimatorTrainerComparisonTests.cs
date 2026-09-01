using System.IO;
using System.Linq;
using ChileMining.Core.Generation;
using ChileMining.Core.Ml;
using Xunit;

namespace ChileMining.Core.Tests;

public class GradeEstimatorTrainerComparisonTests
{
    [Fact]
    public void Compare_ReturnsAllThreeTrainers_WithReasonableFastTreeFit()
    {
        var samples = SyntheticDataGenerator.GenerateDrillHoles(count: 2000, seed: 42);

        var results = GradeEstimatorTrainerComparison.Compare(samples);

        Assert.Equal(3, results.Count);
        Assert.Contains(results, r => r.TrainerName == "FastTree");
        Assert.Contains(results, r => r.TrainerName == "SDCA");
        Assert.Contains(results, r => r.TrainerName == "OnlineGradientDescent");

        // FastTree ya se sabe (GradeEstimatorTests) que aprende la señal real
        // en este dataset -- si ese mismo trainer, corrido a traves de esta
        // comparacion (mismo pipeline y split), no llega a un R2 razonable,
        // el pipeline compartido de la comparacion tiene un bug real.
        var fastTree = results.Single(r => r.TrainerName == "FastTree");
        Assert.True(fastTree.Metrics.RSquared > 0.5, $"R-cuadrado de FastTree demasiado bajo: {fastTree.Metrics.RSquared:F3}");
    }

    [Fact]
    public void SaveToCsv_WritesOneRowPerTrainer_UsingInvariantCulture()
    {
        var samples = SyntheticDataGenerator.GenerateDrillHoles(count: 500, seed: 42);
        var results = GradeEstimatorTrainerComparison.Compare(samples);
        string path = Path.Combine(Path.GetTempPath(), $"grade_trainer_comparison_{System.Guid.NewGuid():N}.csv");

        try
        {
            GradeEstimatorTrainerComparison.SaveToCsv(results, path);

            var lines = File.ReadAllLines(path);
            Assert.Equal("Trainer,RSquared,RMSE,MAE", lines[0]);
            Assert.Equal(4, lines.Length); // header + 3 trainers
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}
