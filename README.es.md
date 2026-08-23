<div align="center">

# ⛏️ Chile Mining -- Control de Leyes y Calidad de Tronadura (.NET)

**Una solucion de ciencia de datos nativa en C# / ML.NET para estimacion de ley de cobre y calidad de fragmentacion de tronadura, construida y abierta directamente en Visual Studio**

🌐 **[English](README.md)** | **[Español](README.es.md)**

[![.NET 8](https://img.shields.io/badge/.NET-8.0-512BD4)](https://dotnet.microsoft.com/)
[![ML.NET](https://img.shields.io/badge/ML.NET-5.0-4285F4)](https://dotnet.microsoft.com/apps/ai/ml-dotnet)
[![WPF](https://img.shields.io/badge/UI-WPF-0078D7)](https://learn.microsoft.com/dotnet/desktop/wpf/)
[![xUnit](https://img.shields.io/badge/tests-7%20passing-brightgreen)](tests/ChileMining.Core.Tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

</div>

---

## 1. Por qué existe este proyecto

Todos los proyectos anteriores de este portafolio (asistente RAG, mantención predictiva, optimización de flotación, el data warehouse dbt/DuckDB) fueron Python de punta a punta -- incluso el warehouse dbt, que agregó una capa real de modelado SQL, sigue leyéndose como mayoritariamente Python en la barra de lenguajes de GitHub al comparar bytes. Este proyecto es un giro deliberado: **todo el pipeline -- generación de datos, entrenamiento ML, evaluación, y la app interactiva -- es C#, construido como una solución real multi-proyecto de Visual Studio (`.sln` + 4 `.csproj`), no un script único.**

## 2. Problema de negocio

Los equipos de planificación minera y geología toman constantemente dos decisiones durante el control de leyes y el diseño de tronadura:

1. **Control de leyes**: dadas las lecturas geoquímicas/geofísicas de un sondaje, ¿este material es mineral o estéril? El software geoestadístico manual suele ser lento de iterar en terreno.
2. **QA de diseño de tronadura**: dado el burden, espaciamiento, factor de potencia, dureza de roca y diámetro de perforación de una tronadura, ¿la fragmentación resultante será lo suficientemente fina para evitar re-tronaduras costosas o daño al chancador por sobre-tamaño?

Este proyecto construye dos modelos ML.NET -- una **regresión** para ley de cobre y un **clasificador multiclase** para calidad de fragmentación -- servidos a través de una **app de escritorio WPF** nativa para uso interactivo, sin necesidad de navegador.

## 3. Estructura de la solución

```
chile-mining-grade-blast-quality-dotnet/
├── ChileMining.sln
├── src/
│   ├── ChileMining.Core/                  # librería de clases: modelos de datos, generador, pipelines ML.NET
│   │   ├── Data/                          # DrillHoleSample, BlastDesign
│   │   ├── Generation/                    # SyntheticDataGenerator
│   │   └── Ml/                            # GradeEstimator, FragmentationClassifier
│   ├── ChileMining.Trainer/               # app de consola: generar -> entrenar -> evaluar -> guardar
│   └── ChileMining.DesktopApp/            # app WPF: asistente interactivo de leyes y tronadura
├── tests/
│   └── ChileMining.Core.Tests/            # xUnit: 7 tests
├── data/                                  # CSVs + modelos .zip entrenados (generado, en .gitignore)
├── README.md
└── README.es.md
```

Abre `ChileMining.sln` directamente en Visual Studio -- la solución, referencias entre proyectos, y paquetes NuGet (`Microsoft.ML`, `Microsoft.ML.FastTree`) ya están conectados y listos para compilar con F5.

## 4. Las dos tareas de ML

**Estimación de ley (regresión, FastTree)** -- predice la ley de cobre (%) desde `ProfundidadM`, `UnidadGeologica`, `TipoAlteracion`, `DensidadGrCm3`, `ResistividadOhmM`, `DistanciaFallaM`. El generador sintético liga la ley a la geología tal como zonan los depósitos porfídicos de cobre reales: los núcleos de alteración potásica en pórfido/skarn obtienen la mayor ley base, la andesita con alteración propilítica la menor, con densidad y resistividad correlacionadas a la ley mediante relaciones físicas plausibles (más sulfuros → roca más densa, menor resistividad) -- no ruido aleatorio independiente.

**Fragmentación de tronadura (multiclase, SDCA)** -- predice `Fino` / `Medio` / `Grueso` / `SobreTamano` desde `BurdenM`, `EspaciamientoM`, `FactorPotenciaKgTon`, `DurezaRocaMpa`, `DiametroPerforacionMm`, usando un índice de fragmentación estilo Kuz-Ram (mayor factor de potencia y malla más cerrada → fragmentación más fina; roca más dura → más gruesa) para asignar las etiquetas antes de agregar ruido -- de modo que, igual que en el modelo de leyes, la etiqueta está causalmente fundamentada en las features en vez de muestreada independientemente de ellas.

## 5. Instalación

Requiere el [SDK de .NET 8](https://dotnet.microsoft.com/download) (o Visual Studio 2022+ con la carga de trabajo "Desarrollo de escritorio .NET", que ya lo incluye).

```powershell
git clone https://github.com/Rxyxs/chile-mining-grade-blast-quality-dotnet.git
cd chile-mining-grade-blast-quality-dotnet
dotnet restore
```

## 6. Uso

**1. Generar datos, entrenar y evaluar ambos modelos:**

```powershell
dotnet run --project src/ChileMining.Trainer
```

Escribe `drill_holes.csv`, `blast_designs.csv`, `grade_estimator.zip`, y `fragmentation_classifier.zip` en `data/`.

**2. Levantar la app de escritorio** (después de que el paso 1 haya corrido al menos una vez):

```powershell
dotnet run --project src/ChileMining.DesktopApp
```

Dos pestañas: **Control de Leyes** (ingresa parámetros de sondaje, obtén una ley estimada + clasificación mineral/estéril contra una ley de corte ilustrativa) y **Diseño de Tronadura** (ingresa parámetros de diseño de tronadura, obtén una categoría de fragmentación predicha).

**3. Correr los tests:**

```powershell
dotnet test
```

O abre `ChileMining.sln` en Visual Studio y usa Test Explorer / F5 directamente.

## 7. Resultados validados

Todos los números a continuación provienen de ejecutar realmente `ChileMining.Trainer` en este repositorio:

| Métrica | Valor |
|---|---|
| Muestras de sondaje generadas | 2.000 |
| Muestras de diseño de tronadura generadas | 2.000 |
| Estimador de ley -- R² | **0,833** |
| Estimador de ley -- RMSE | 0,121 (unidades de ley, es decir ±0,12 puntos porcentuales de Cu%) |
| Estimador de ley -- MAE | 0,096 |
| Clasificador de fragmentación -- MicroAccuracy | **0,823** (vs. ~0,25 de línea base aleatoria para 4 clases) |
| Clasificador de fragmentación -- MacroAccuracy | 0,798 |
| Clasificador de fragmentación -- LogLoss | 0,431 |
| Tests xUnit | **7/7 pasando** |

Dos de los tests xUnit protegen específicamente contra una clase de bug ya encontrada antes en este portafolio: un clasificador cuya etiqueta en realidad no está correlacionada con sus features se ve bien hasta que revisas las métricas y encuentras un desempeño casi aleatorio. Aquí, `PotasicaAlteration_HasHigherAverageGrade_ThanPropilitica` y `HigherPowderFactor_ProducesFinerFragmentation_OnAverage` verifican directamente la relación causal en el generador, y `TrainAndEvaluate_LearnsRealSignal_*` verifican que los modelos entrenados superen un umbral de señal real, no solo "el código corre".

## 8. Un bug de formato cultural que vale la pena conocer

Los escritores de CSV de `SyntheticDataGenerator` usan `FormattableString.Invariant(...)` en cada campo numérico. Sin esto, en una máquina con configuración regional en español (Chile), `$"{valor}"` formatea los floats con **coma** como separador decimal (`50,5`) -- lo que corrompe el CSV, ya que la coma también es el delimitador de columnas. Esto está cubierto por un test de regresión dedicado (`SaveDrillHolesToCsv_UsesInvariantCulture_RegardlessOfSystemCulture`) que cambia temporalmente `CurrentCulture` a `es-CL` y verifica que el archivo siga interpretándose como 7 columnas por fila. La app WPF toma el enfoque opuesto, deliberadamente, para *texto de cara al usuario*: parsea la entrada con `CurrentCulture` primero (para que un usuario chileno pueda escribir `0,30` de forma natural) y cae a `InvariantCulture` como respaldo (para que los valores por defecto del XAML, escritos con punto, sigan funcionando) -- el I/O de archivos quiere portabilidad, el texto de UI quiere coincidir con lo que el usuario realmente escribió.

## 9. Disclaimer

Todos los datos son sintéticos, generados por `SyntheticDataGenerator` con una semilla fija. La ley de corte usada en la app de escritorio (0,30% Cu) es ilustrativa, no una ley de corte económica real -- una ley de corte real depende de los costos de mina/planta y del precio del cobre, y no es una constante fija.

## Licencia

MIT -- ver [LICENSE](LICENSE).

## Autor

**Pablo Reyes** -- [github.com/Rxyxs](https://github.com/Rxyxs)
