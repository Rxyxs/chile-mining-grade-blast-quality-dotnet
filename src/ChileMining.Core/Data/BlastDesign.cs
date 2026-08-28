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
    public float AlturaBancoM { get; set; }

    /// <summary>
    /// P80: tamano de malla de tamiz (cm) bajo el cual pasa el 80% de la masa
    /// fragmentada -- el KPI estandar de fragmentacion de tronadura en mineria
    /// (no un indice propio del proyecto). Calculado en SyntheticDataGenerator
    /// con el modelo Kuznetsov + distribucion Rosin-Rammler, el estandar de la
    /// industria para predecir fragmentacion a partir del diseno de la malla.
    /// </summary>
    [LoadColumn(6)]
    [ColumnName("Label")]
    public float P80Cm { get; set; }

    [LoadColumn(7)]
    [ColumnName("FragmentationLabel")]
    public string CalidadFragmentacion { get; set; } = string.Empty;
}

public class FragmentationPrediction
{
    [ColumnName("PredictedLabel")]
    public string CalidadPredicha { get; set; } = string.Empty;

    public float[] Score { get; set; } = System.Array.Empty<float>();
}

public class P80Prediction
{
    [ColumnName("Score")]
    public float P80CmEstimado { get; set; }
}
