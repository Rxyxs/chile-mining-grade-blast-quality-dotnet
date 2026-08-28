namespace ChileMining.Core.Ml;

/// <summary>
/// Bucketiza un P80 (cm) en una categoria operativa de calidad de
/// fragmentacion. Umbrales ilustrativos del proyecto (no un estandar
/// SERNAGEOMIN/industria), calibrados empiricamente contra la distribucion
/// real de P80 que produce el modelo Kuz-Ram/Rosin-Rammler de
/// SyntheticDataGenerator para el rango de parametros de malla usado.
/// Unica fuente de verdad para estos umbrales -- tanto el generador de
/// datos sinteticos (etiqueta de entrenamiento) como el CLI de inferencia
/// (clasificacion de una prediccion nueva) llaman a este mismo metodo, para
/// que no puedan divergir silenciosamente.
/// </summary>
public static class FragmentationQuality
{
    public static string Classify(float p80Cm) => p80Cm switch
    {
        <= 30f => "Fino",
        <= 45f => "Medio",
        <= 60f => "Grueso",
        _ => "SobreTamano",
    };
}
