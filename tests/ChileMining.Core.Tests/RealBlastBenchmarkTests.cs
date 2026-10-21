using System;
using System.IO;
using System.Linq;
using ChileMining.Core.Ml;
using Xunit;

namespace ChileMining.Core.Tests;

public class RealBlastBenchmarkTests
{
    private static string RealDir()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "ChileMining.sln")))
        {
            dir = dir.Parent;
        }

        return Path.Combine(dir!.FullName, "data", "real");
    }

    [Fact]
    public void PublishedCorpus_Has97BlastsFromTenSites()
    {
        var corpus = RealBlastBenchmark.LoadCsv(Path.Combine(RealDir(), "hudaverdi1997.csv"));
        Assert.Equal(97, corpus.Count);
        Assert.Equal(10, corpus.Select(b => b.Site).Distinct().Count());
        Assert.All(corpus, b => Assert.InRange(b.X50M, 0.01f, 1.0f));
    }

    [Fact]
    public void Holdout_CarriesThePublishedKuzRamPredictions()
    {
        var holdout = RealBlastBenchmark.LoadCsv(Path.Combine(RealDir(), "hudaverdi_holdout.csv"));
        Assert.Equal(14, holdout.Count);
        Assert.Equal(13, holdout.Count(b => b.PublishedKuzRamM.HasValue));
    }

    [Fact]
    public void VarianceExplained_IsZeroForTheMeanAndNegativeWhenWorse()
    {
        double[] actual = { 1, 2, 3, 4 };
        Assert.Equal(1.0, RealBlastBenchmark.VarianceExplained(actual, actual), 9);
        Assert.Equal(0.0, RealBlastBenchmark.VarianceExplained(actual, new double[] { 2.5, 2.5, 2.5, 2.5 }), 9);
        Assert.True(RealBlastBenchmark.VarianceExplained(actual, new double[] { 4, 3, 2, 1 }) < 0);
    }
}
