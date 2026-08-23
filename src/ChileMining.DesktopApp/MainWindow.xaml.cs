using System;
using System.Globalization;
using System.IO;
using System.Windows;
using ChileMining.Core.Data;
using ChileMining.Core.Ml;

namespace ChileMining.DesktopApp;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window
{
    // Ley de corte ilustrativa para clasificar Mineral/Esteril -- valor de ejemplo,
    // no un corte economico real (eso depende de costos de mina/planta/precio del cobre).
    private const float LeyDeCorteCuPct = 0.30f;

    private GradeEstimator? _gradeEstimator;
    private FragmentationClassifier? _fragmentationClassifier;
    private string? _modelLoadError;

    public MainWindow()
    {
        InitializeComponent();
        LoadModels();
    }

    private void LoadModels()
    {
        try
        {
            string dataDir = Path.Combine(FindRepoRoot(), "data");
            string gradeModelPath = Path.Combine(dataDir, "grade_estimator.zip");
            string fragmentationModelPath = Path.Combine(dataDir, "fragmentation_classifier.zip");

            if (!File.Exists(gradeModelPath) || !File.Exists(fragmentationModelPath))
            {
                _modelLoadError =
                    "No se encontraron los modelos entrenados en data/. Corre primero:\n" +
                    "dotnet run --project src/ChileMining.Trainer";
                return;
            }

            _gradeEstimator = new GradeEstimator();
            _gradeEstimator.Load(gradeModelPath);

            _fragmentationClassifier = new FragmentationClassifier();
            _fragmentationClassifier.Load(fragmentationModelPath);
        }
        catch (Exception ex)
        {
            _modelLoadError = $"Error cargando los modelos: {ex.Message}";
        }
    }

    private void EstimarLeyButton_Click(object sender, RoutedEventArgs e)
    {
        if (_gradeEstimator is null)
        {
            ResultadoLeyTextBlock.Text = _modelLoadError ?? "El modelo de leyes no esta disponible.";
            return;
        }

        if (!TryParseFloat(ProfundidadTextBox.Text, out float profundidad) ||
            !TryParseFloat(DensidadTextBox.Text, out float densidad) ||
            !TryParseFloat(ResistividadTextBox.Text, out float resistividad) ||
            !TryParseFloat(DistanciaFallaTextBox.Text, out float distanciaFalla))
        {
            ResultadoLeyTextBlock.Text = "Revisa los valores numericos ingresados -- alguno no se pudo interpretar.";
            return;
        }

        var muestra = new DrillHoleSample
        {
            ProfundidadM = profundidad,
            UnidadGeologica = SelectedText(UnidadGeologicaComboBox),
            TipoAlteracion = SelectedText(TipoAlteracionComboBox),
            DensidadGrCm3 = densidad,
            ResistividadOhmM = resistividad,
            DistanciaFallaM = distanciaFalla,
        };

        var prediccion = _gradeEstimator.Predict(muestra);
        string clasificacion = prediccion.LeyCuPctEstimada >= LeyDeCorteCuPct ? "Mineral" : "Esteril";

        ResultadoLeyTextBlock.Text = string.Format(
            CultureInfo.CurrentCulture,
            "Ley de Cu estimada: {0:F3}%\nClasificacion: {1} (ley de corte ilustrativa: {2:F2}%)",
            prediccion.LeyCuPctEstimada * 100f,
            clasificacion,
            LeyDeCorteCuPct * 100f);
    }

    private void PredecirFragmentacionButton_Click(object sender, RoutedEventArgs e)
    {
        if (_fragmentationClassifier is null)
        {
            ResultadoFragmentacionTextBlock.Text = _modelLoadError ?? "El modelo de fragmentacion no esta disponible.";
            return;
        }

        if (!TryParseFloat(BurdenTextBox.Text, out float burden) ||
            !TryParseFloat(EspaciamientoTextBox.Text, out float espaciamiento) ||
            !TryParseFloat(FactorPotenciaTextBox.Text, out float factorPotencia) ||
            !TryParseFloat(DurezaRocaTextBox.Text, out float dureza) ||
            !TryParseFloat(DiametroTextBox.Text, out float diametro))
        {
            ResultadoFragmentacionTextBlock.Text = "Revisa los valores numericos ingresados -- alguno no se pudo interpretar.";
            return;
        }

        var diseno = new BlastDesign
        {
            BurdenM = burden,
            EspaciamientoM = espaciamiento,
            FactorPotenciaKgTon = factorPotencia,
            DurezaRocaMpa = dureza,
            DiametroPerforacionMm = diametro,
        };

        var prediccion = _fragmentationClassifier.Predict(diseno);

        ResultadoFragmentacionTextBlock.Text = string.Format(
            CultureInfo.CurrentCulture,
            "Calidad de fragmentacion predicha: {0}",
            prediccion.CalidadPredicha);
    }

    private static string SelectedText(System.Windows.Controls.ComboBox comboBox)
    {
        return (comboBox.SelectedItem as System.Windows.Controls.ComboBoxItem)?.Content?.ToString()
            ?? string.Empty;
    }

    // Intenta primero con la cultura actual (para que un usuario en Chile pueda
    // escribir "0,30" de forma natural) y cae a cultura invariante como respaldo
    // (cubre los valores por defecto del XAML, escritos con punto decimal).
    private static bool TryParseFloat(string text, out float value)
    {
        if (float.TryParse(text, NumberStyles.Float, CultureInfo.CurrentCulture, out value))
        {
            return true;
        }

        return float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out value);
    }

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "ChileMining.sln")))
        {
            dir = dir.Parent;
        }

        return dir?.FullName
            ?? throw new DirectoryNotFoundException("No se encontro ChileMining.sln en ningun directorio padre.");
    }
}
