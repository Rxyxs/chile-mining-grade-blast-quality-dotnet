using Microsoft.ML.Data;

namespace ChileMining.Core.Data;

/// <summary>
/// Parametros de diseno de una tronadura y la calidad de fragmentacion resultante.
/// Los datos son sinteticos, pero la relacion entre factor de potencia/burden/espaciamiento/
/// dureza de roca y fragmentacion sigue la intuicion fisica estandar de voladura
/// (mayor factor de potencia y malla mas cerrada -> fragmentacion mas fina).
/// </summary>
public class BlastDesign
{
    [LoadColumn(0)]
    public float BurdenM { get; set; }

    [LoadColumn(1)]
    public float EspaciamientoM { get; set; }

    [LoadColumn(2)]
    public float FactorPotenciaKgTon { get; set; }

    [LoadColumn(3)]
    public float DurezaRocaMpa { get; set; }

    [LoadColumn(4)]
    public float DiametroPerforacionMm { get; set; }

    [LoadColumn(5)]
    [ColumnName("Label")]
    public string CalidadFragmentacion { get; set; } = string.Empty;
}

public class FragmentationPrediction
{
    [ColumnName("PredictedLabel")]
    public string CalidadPredicha { get; set; } = string.Empty;

    public float[] Score { get; set; } = System.Array.Empty<float>();
}
