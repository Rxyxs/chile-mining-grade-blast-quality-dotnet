using Microsoft.ML.Data;

namespace ChileMining.Core.Data;

/// <summary>
/// Muestra geoquimica de un tramo de sondaje, usada para estimar la ley de cobre.
/// Los datos son 100% sinteticos (ver Generation/SyntheticDataGenerator), pero las
/// correlaciones entre unidad geologica/alteracion y ley reflejan zonacion real de
/// porfidos cupriferos (nucleo potasico de mayor ley -> propilitica distal de menor ley).
/// </summary>
public class DrillHoleSample
{
    [LoadColumn(0)]
    public float ProfundidadM { get; set; }

    [LoadColumn(1)]
    public string UnidadGeologica { get; set; } = string.Empty;

    [LoadColumn(2)]
    public string TipoAlteracion { get; set; } = string.Empty;

    [LoadColumn(3)]
    public float DensidadGrCm3 { get; set; }

    [LoadColumn(4)]
    public float ResistividadOhmM { get; set; }

    [LoadColumn(5)]
    public float DistanciaFallaM { get; set; }

    [LoadColumn(6)]
    [ColumnName("Label")]
    public float LeyCuPct { get; set; }
}

public class GradePrediction
{
    [ColumnName("Score")]
    public float LeyCuPctEstimada { get; set; }
}
