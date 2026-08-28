using System.Globalization;
using ChileMining.Core.Data;
using ChileMining.Core.Ml;

namespace ChileMining.Cli;

/// <summary>
/// CLI: lee un archivo CSV de mallas de tronadura (diseno de perforacion/carga,
/// sin fragmentacion conocida todavia) y predice P80 (cm) por fila usando el
/// modelo ONNX exportado por ChileMining.Trainer -- inferencia via
/// Microsoft.ML.OnnxRuntime directamente, sin depender del runtime de ML.NET.
///
/// Uso:
///   chilemining-cli --input mallas.csv --onnx data/p80_estimator.onnx [--output resultado.csv]
///
/// CSV de entrada esperado (con encabezado, en cualquier orden de columnas):
///   BurdenM,EspaciamientoM,FactorPotenciaKgTon,DurezaRocaMpa,DiametroPerforacionMm,AlturaBancoM
/// </summary>
public static class Program
{
    private static readonly string[] RequiredColumns =
    {
        "BurdenM", "EspaciamientoM", "FactorPotenciaKgTon", "DurezaRocaMpa", "DiametroPerforacionMm", "AlturaBancoM",
    };

    public static int Main(string[] args)
    {
        string? inputPath = null;
        string? onnxPath = null;
        string? outputPath = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--input": inputPath = args[++i]; break;
                case "--onnx": onnxPath = args[++i]; break;
                case "--output": outputPath = args[++i]; break;
                case "--help": PrintHelp(); return 0;
                default:
                    Console.Error.WriteLine($"Argumento desconocido: {args[i]}");
                    PrintHelp();
                    return 1;
            }
        }

        if (inputPath is null || onnxPath is null)
        {
            Console.Error.WriteLine("Error: --input y --onnx son obligatorios.\n");
            PrintHelp();
            return 1;
        }

        if (!File.Exists(inputPath))
        {
            Console.Error.WriteLine($"Error: no se encontro el archivo de entrada: {inputPath}");
            return 1;
        }

        if (!File.Exists(onnxPath))
        {
            Console.Error.WriteLine($"Error: no se encontro el modelo ONNX: {onnxPath}\n"
                + "Genera uno primero con: dotnet run --project src/ChileMining.Trainer");
            return 1;
        }

        List<(BlastDesign Design, int RowNumber)> rows;
        try
        {
            rows = ReadBlastDesignsCsv(inputPath);
        }
        catch (FormatException ex)
        {
            Console.Error.WriteLine($"Error leyendo {inputPath}: {ex.Message}");
            return 1;
        }

        using var inference = new OnnxP80InferenceService(onnxPath);

        var results = new List<(BlastDesign Design, float P80, string Calidad)>();
        foreach (var (design, _) in rows)
        {
            float p80 = inference.PredictP80Cm(design);
            results.Add((design, p80, FragmentationQuality.Classify(p80)));
        }

        Console.WriteLine($"{"Fila",-6}{"Burden",-9}{"Espac.",-9}{"FP kg/t",-10}{"Dureza",-9}{"Diam mm",-10}{"P80 cm",-9}Calidad");
        for (int i = 0; i < results.Count; i++)
        {
            var (d, p80, calidad) = results[i];
            Console.WriteLine(FormattableString.Invariant(
                $"{i + 1,-6}{d.BurdenM,-9:F2}{d.EspaciamientoM,-9:F2}{d.FactorPotenciaKgTon,-10:F3}{d.DurezaRocaMpa,-9:F1}{d.DiametroPerforacionMm,-10:F1}{p80,-9:F1}{calidad}"));
        }

        int sobreTamano = results.Count(r => r.Calidad == "SobreTamano");
        Console.WriteLine($"\n{results.Count} mallas procesadas, {sobreTamano} con riesgo de sobretamano (requieren revision de diseno).");

        if (outputPath is not null)
        {
            WriteResultsCsv(results, outputPath);
            Console.WriteLine($"Resultado escrito en: {outputPath}");
        }

        return 0;
    }

    private static List<(BlastDesign, int)> ReadBlastDesignsCsv(string path)
    {
        var lines = File.ReadAllLines(path);
        if (lines.Length == 0)
        {
            throw new FormatException("El archivo esta vacio.");
        }

        string[] header = lines[0].Split(',').Select(h => h.Trim()).ToArray();
        var columnIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < header.Length; i++) columnIndex[header[i]] = i;

        foreach (string required in RequiredColumns)
        {
            if (!columnIndex.ContainsKey(required))
            {
                throw new FormatException(
                    $"Falta la columna requerida '{required}'. Columnas esperadas: {string.Join(", ", RequiredColumns)}");
            }
        }

        float Col(string[] fields, string name) =>
            float.Parse(fields[columnIndex[name]], NumberStyles.Float, CultureInfo.InvariantCulture);

        var rows = new List<(BlastDesign, int)>();
        for (int lineNum = 1; lineNum < lines.Length; lineNum++)
        {
            if (string.IsNullOrWhiteSpace(lines[lineNum])) continue;

            string[] fields = lines[lineNum].Split(',');
            try
            {
                var design = new BlastDesign
                {
                    BurdenM = Col(fields, "BurdenM"),
                    EspaciamientoM = Col(fields, "EspaciamientoM"),
                    FactorPotenciaKgTon = Col(fields, "FactorPotenciaKgTon"),
                    DurezaRocaMpa = Col(fields, "DurezaRocaMpa"),
                    DiametroPerforacionMm = Col(fields, "DiametroPerforacionMm"),
                    AlturaBancoM = Col(fields, "AlturaBancoM"),
                };
                rows.Add((design, lineNum + 1));
            }
            catch (Exception ex) when (ex is FormatException or IndexOutOfRangeException)
            {
                throw new FormatException($"Fila {lineNum + 1}: no se pudo parsear como numero valido ({ex.Message})");
            }
        }

        return rows;
    }

    private static void WriteResultsCsv(List<(BlastDesign Design, float P80, string Calidad)> results, string path)
    {
        using var writer = new StreamWriter(path, false);
        writer.WriteLine("BurdenM,EspaciamientoM,FactorPotenciaKgTon,DurezaRocaMpa,DiametroPerforacionMm,AlturaBancoM,P80EstimadoCm,CalidadEstimada");
        foreach (var (d, p80, calidad) in results)
        {
            writer.WriteLine(FormattableString.Invariant(
                $"{d.BurdenM},{d.EspaciamientoM},{d.FactorPotenciaKgTon},{d.DurezaRocaMpa},{d.DiametroPerforacionMm},{d.AlturaBancoM},{p80:F2},{calidad}"));
        }
    }

    private static void PrintHelp()
    {
        Console.WriteLine(
            "chilemining-cli -- prediccion de P80 de fragmentacion para mallas de tronadura\n\n"
            + "Uso:\n"
            + "  chilemining-cli --input <mallas.csv> --onnx <modelo.onnx> [--output <resultado.csv>]\n\n"
            + "CSV de entrada (con encabezado): " + string.Join(",", RequiredColumns));
    }
}
