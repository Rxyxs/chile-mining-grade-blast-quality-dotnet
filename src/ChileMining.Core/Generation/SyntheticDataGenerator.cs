using System;
using System.Collections.Generic;
using System.IO;
using ChileMining.Core.Data;

namespace ChileMining.Core.Generation;

/// <summary>
/// Genera datos sinteticos pero fisicamente consistentes para los dos problemas del
/// proyecto: control de leyes (regresion) y calidad de fragmentacion (clasificacion).
/// En ambos casos la etiqueta esta causalmente ligada a las features generadas -- no
/// es un valor aleatorio independiente -- para que los modelos de ML.NET tengan una
/// senal real que aprender.
/// </summary>
public static class SyntheticDataGenerator
{
    private sealed record UnidadAlteracion(string Unidad, string Alteracion, float LeyBaseCuPct, float ResistividadBaseOhmM);

    private static readonly UnidadAlteracion[] Combinaciones =
    {
        new("Porfido Cu-Mo", "Potasica", 0.90f, 40f),
        new("Porfido Cu-Mo", "Filica", 0.60f, 60f),
        new("Skarn", "Potasica", 1.10f, 35f),
        new("Skarn", "Propilitica", 0.40f, 120f),
        new("Brecha Hidrotermal", "Filica", 0.70f, 55f),
        new("Andesita Alterada", "Propilitica", 0.25f, 150f),
        new("Andesita Alterada", "Argilica", 0.35f, 90f),
    };

    public static List<DrillHoleSample> GenerateDrillHoles(int count, int seed = 42)
    {
        var rng = new Random(seed);
        var samples = new List<DrillHoleSample>(count);

        for (int i = 0; i < count; i++)
        {
            var combo = Combinaciones[rng.Next(Combinaciones.Length)];
            float profundidad = (float)(50 + rng.NextDouble() * 450); // 50-500 m
            float ley = MathF.Max(0.02f, combo.LeyBaseCuPct + SampleGaussian(rng, 0f, 0.12f));

            // La densidad sube levemente con la ley (mayor contenido de sulfuros -> mas densidad).
            float densidad = 2.60f + ley * 0.15f + SampleGaussian(rng, 0f, 0.05f);

            // La resistividad baja con mayor mineralizacion/alteracion (sulfuros y arcillas conducen mas).
            float resistividad = MathF.Max(5f, combo.ResistividadBaseOhmM - ley * 20f + SampleGaussian(rng, 0f, 10f));

            float distanciaFalla = (float)(rng.NextDouble() * 300); // 0-300 m

            samples.Add(new DrillHoleSample
            {
                ProfundidadM = MathF.Round(profundidad, 1),
                UnidadGeologica = combo.Unidad,
                TipoAlteracion = combo.Alteracion,
                DensidadGrCm3 = MathF.Round(densidad, 3),
                ResistividadOhmM = MathF.Round(resistividad, 2),
                DistanciaFallaM = MathF.Round(distanciaFalla, 1),
                LeyCuPct = MathF.Round(ley, 4),
            });
        }

        return samples;
    }

    // Roca-tipo huesped de porfido cuprifero: no varia por diseno de malla, es
    // una propiedad del macizo rocoso -- se fija como constante en vez de
    // agregarse como feature, igual que en la practica de ingenieria de
    // tronadura (se mide una vez por dominio geotecnico, no por malla).
    private const float DensidadRocaTonM3 = 2.65f;

    // Relative Weight Strength del explosivo vs. ANFO (ANFO = 100 por
    // definicion) -- simplificacion documentada: el proyecto no modela
    // distintos tipos de explosivo, solo ANFO-equivalente.
    private const float RwsAnfo = 100f;

    public static List<BlastDesign> GenerateBlastDesigns(int count, int seed = 43)
    {
        var rng = new Random(seed);
        var designs = new List<BlastDesign>(count);

        for (int i = 0; i < count; i++)
        {
            float burden = (float)(3 + rng.NextDouble() * 5);           // 3-8 m
            float espaciamiento = (float)(4 + rng.NextDouble() * 6);    // 4-10 m
            float factorPotencia = (float)(0.15 + rng.NextDouble() * 0.30); // 0.15-0.45 kg/ton
            float dureza = (float)(50 + rng.NextDouble() * 200);        // 50-250 MPa
            float diametro = (float)(150 + rng.NextDouble() * 160);     // 150-310 mm
            float alturaBanco = (float)(10 + rng.NextDouble() * 5);     // 10-15 m, banco tipico rajo abierto

            float p80 = ComputeP80Cm(burden, espaciamiento, factorPotencia, dureza, diametro, alturaBanco, rng);
            string calidad = ChileMining.Core.Ml.FragmentationQuality.Classify(p80);

            designs.Add(new BlastDesign
            {
                BurdenM = MathF.Round(burden, 2),
                EspaciamientoM = MathF.Round(espaciamiento, 2),
                FactorPotenciaKgTon = MathF.Round(factorPotencia, 3),
                DurezaRocaMpa = MathF.Round(dureza, 1),
                DiametroPerforacionMm = MathF.Round(diametro, 1),
                AlturaBancoM = MathF.Round(alturaBanco, 2),
                P80Cm = MathF.Round(p80, 2),
                CalidadFragmentacion = calidad,
            });
        }

        return designs;
    }

    /// <summary>
    /// P80 (cm) via el modelo Kuznetsov (tamano medio de fragmento) + distribucion
    /// Rosin-Rammler (percentil 80 de esa distribucion) -- el estandar de la
    /// industria de voladura para predecir fragmentacion a partir del diseno de
    /// malla, no una formula propia del proyecto. Referencia: Cunningham (1987),
    /// "Fragmentation Estimation and the Kuz-Ram Model -- Four Years On".
    ///
    /// Simplificacion documentada: el indice de uniformidad de Cunningham normalmente
    /// incluye terminos de largo de taco/carga (W, L) que este dataset no modela
    /// explicitamente (no hay diseno de carga por deck); se omiten esos dos factores,
    /// manteniendo burden/espaciamiento/diametro que si estan disponibles.
    /// </summary>
    private static float ComputeP80Cm(
        float burdenM, float espaciamientoM, float factorPotenciaKgTon,
        float durezaRocaMpa, float diametroPerforacionMm, float alturaBancoM, Random rng)
    {
        // Factor de roca A de Kuznetsov: mapeo lineal de dureza (MPa) al rango
        // publicado A in [7, 13] (roca blanda/muy fracturada -> dura/masiva).
        float rockFactorA = Math.Clamp(4f + durezaRocaMpa * 0.036f, 7f, 13f);

        float volumenPorTiroM3 = burdenM * espaciamientoM * alturaBancoM;
        float masaExplosivoKg = MathF.Max(0.01f, factorPotenciaKgTon * volumenPorTiroM3 * DensidadRocaTonM3);

        float tamanoMedioCm = rockFactorA
            * MathF.Pow(volumenPorTiroM3 / masaExplosivoKg, 0.8f)
            * MathF.Pow(masaExplosivoKg, 1f / 6f)
            * MathF.Pow(115f / RwsAnfo, 19f / 30f);

        // Indice de uniformidad n de Cunningham (simplificado, ver docstring).
        float n = (2.2f - 14f * (burdenM / diametroPerforacionMm))
                  * MathF.Sqrt((1f + espaciamientoM / burdenM) / 2f);
        n = Math.Clamp(n, 0.6f, 2.2f);

        // Rosin-Rammler: tamano bajo el cual pasa una fraccion P de la masa.
        float p80 = tamanoMedioCm * MathF.Pow(MathF.Log(1f / (1f - 0.8f)), 1f / n);

        return MathF.Max(1f, p80 + SampleGaussian(rng, 0f, p80 * 0.05f));
    }

    private static float SampleGaussian(Random rng, float mean, float stdDev)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * (float)randStdNormal;
    }

    // FormattableString.Invariant fuerza CultureInfo.InvariantCulture en todo el string
    // interpolado -- sin esto, en un SO con configuracion regional es-CL los numeros se
    // formatean con coma decimal (p.ej. "50,5"), lo que rompe el CSV (la coma tambien es
    // el separador de columnas).
    public static void SaveDrillHolesToCsv(IEnumerable<DrillHoleSample> samples, string path)
    {
        using var writer = new StreamWriter(path, false);
        writer.WriteLine("ProfundidadM,UnidadGeologica,TipoAlteracion,DensidadGrCm3,ResistividadOhmM,DistanciaFallaM,LeyCuPct");
        foreach (var s in samples)
        {
            writer.WriteLine(FormattableString.Invariant(
                $"{s.ProfundidadM},{s.UnidadGeologica},{s.TipoAlteracion},{s.DensidadGrCm3},{s.ResistividadOhmM},{s.DistanciaFallaM},{s.LeyCuPct}"));
        }
    }

    public static void SaveBlastDesignsToCsv(IEnumerable<BlastDesign> designs, string path)
    {
        using var writer = new StreamWriter(path, false);
        writer.WriteLine("BurdenM,EspaciamientoM,FactorPotenciaKgTon,DurezaRocaMpa,DiametroPerforacionMm,AlturaBancoM,P80Cm,CalidadFragmentacion");
        foreach (var d in designs)
        {
            writer.WriteLine(FormattableString.Invariant(
                $"{d.BurdenM},{d.EspaciamientoM},{d.FactorPotenciaKgTon},{d.DurezaRocaMpa},{d.DiametroPerforacionMm},{d.AlturaBancoM},{d.P80Cm},{d.CalidadFragmentacion}"));
        }
    }
}
