using System.Globalization;
using System.IO;
using System.Linq;
using ChileMining.Core.Generation;
using Xunit;

namespace ChileMining.Core.Tests;

public class SyntheticDataGeneratorTests
{
    [Fact]
    public void PotasicaAlteration_HasHigherAverageGrade_ThanPropilitica()
    {
        var samples = SyntheticDataGenerator.GenerateDrillHoles(count: 3000, seed: 42);

        double promedioPotasica = samples.Where(s => s.TipoAlteracion == "Potasica").Average(s => s.LeyCuPct);
        double promedioPropilitica = samples.Where(s => s.TipoAlteracion == "Propilitica").Average(s => s.LeyCuPct);

        Assert.True(
            promedioPotasica > promedioPropilitica,
            $"Potasica ({promedioPotasica:F3}) deberia tener mayor ley promedio que Propilitica ({promedioPropilitica:F3})");
    }

    [Fact]
    public void HigherPowderFactor_ProducesFinerFragmentation_OnAverage()
    {
        var designs = SyntheticDataGenerator.GenerateBlastDesigns(count: 3000, seed: 43);
        var ordenado = designs.OrderBy(d => d.FactorPotenciaKgTon).ToList();

        var mitadBaja = ordenado.Take(ordenado.Count / 2);
        var mitadAlta = ordenado.Skip(ordenado.Count / 2);

        double proporcionFinaBaja = mitadBaja.Count(d => d.CalidadFragmentacion == "Fino") / (double)mitadBaja.Count();
        double proporcionFinaAlta = mitadAlta.Count(d => d.CalidadFragmentacion == "Fino") / (double)mitadAlta.Count();

        Assert.True(
            proporcionFinaAlta > proporcionFinaBaja,
            $"Mayor factor de potencia deberia producir mas fragmentacion Fina ({proporcionFinaAlta:P1} vs {proporcionFinaBaja:P1})");
    }

    [Fact]
    public void SaveDrillHolesToCsv_UsesInvariantCulture_RegardlessOfSystemCulture()
    {
        var originalCulture = CultureInfo.CurrentCulture;
        try
        {
            // es-CL formatea decimales con coma ("50,5") -- si SaveDrillHolesToCsv no
            // fuerza cultura invariante, el CSV queda corrupto porque la coma tambien
            // es el separador de columnas.
            CultureInfo.CurrentCulture = new CultureInfo("es-CL");

            var samples = SyntheticDataGenerator.GenerateDrillHoles(count: 5, seed: 42);
            string path = Path.Combine(Path.GetTempPath(), $"test_drill_holes_{System.Guid.NewGuid():N}.csv");
            try
            {
                SyntheticDataGenerator.SaveDrillHolesToCsv(samples, path);
                string[] lines = File.ReadAllLines(path);

                Assert.Equal(6, lines.Length); // 1 header + 5 filas
                foreach (string line in lines.Skip(1))
                {
                    Assert.Equal(7, line.Split(',').Length);
                }
            }
            finally
            {
                File.Delete(path);
            }
        }
        finally
        {
            CultureInfo.CurrentCulture = originalCulture;
        }
    }
}
