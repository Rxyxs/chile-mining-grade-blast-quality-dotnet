<div align="center">

# ⛏️ Chile Mining -- Grade Control & Blast Quality (.NET)

**A native C# / ML.NET data science solution for copper grade estimation and blast fragmentation quality, built and opened directly in Visual Studio**

🌐 **[English](README.md)** | **[Español](README.es.md)**

[![.NET 8](https://img.shields.io/badge/.NET-8.0-512BD4)](https://dotnet.microsoft.com/)
[![ML.NET](https://img.shields.io/badge/ML.NET-5.0-4285F4)](https://dotnet.microsoft.com/apps/ai/ml-dotnet)
[![WPF](https://img.shields.io/badge/UI-WPF-0078D7)](https://learn.microsoft.com/dotnet/desktop/wpf/)
[![xUnit](https://img.shields.io/badge/tests-7%20passing-brightgreen)](tests/ChileMining.Core.Tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

</div>

---

## 1. Motivation

I built this tool around the stack that actually runs day to day in mine planning: native Windows desktop software. Most of the tools geologists and planning engineers use on site (Vulcan, Datamine, Surpac, Deswik) are .NET desktop applications, not notebooks or web dashboards -- and out in the field, with limited connectivity, an executable that runs locally without depending on a server makes a lot more sense than spinning up a browser.

That's why this project is C# end to end -- data generation, ML training and evaluation with ML.NET, and a WPF desktop app -- built as a real Visual Studio solution (`.sln` + 4 `.csproj`), not a loose script. It's also my way of deliberately exercising the .NET side of my data science work: strong typing and strict compilation before an operational decision gets made (is this ore or waste? do I need to re-blast?) is the attitude I want in a tool that informs real site decisions.

## 2. Business problem

Mine planning and geology teams run two decisions constantly during grade control and blast design:

1. **Grade control**: given drill-hole geochemical/geophysical readings, is this material ore or waste? Manual geostatistical software is often slow to iterate with in the field.
2. **Blast design QA**: given a blast's burden, spacing, powder factor, rock hardness, and hole diameter, will fragmentation come out fine enough to avoid costly re-blasting or crusher damage from oversize?

This project builds two ML.NET models -- a **regression** for copper grade and a **multiclass classifier** for fragmentation quality -- served through a native **WPF desktop app** for interactive use, no browser required.

## 3. Solution structure

```
chile-mining-grade-blast-quality-dotnet/
├── ChileMining.sln
├── src/
│   ├── ChileMining.Core/                  # class library: data models, generator, ML.NET pipelines
│   │   ├── Data/                          # DrillHoleSample, BlastDesign
│   │   ├── Generation/                    # SyntheticDataGenerator
│   │   └── Ml/                            # GradeEstimator, FragmentationClassifier
│   ├── ChileMining.Trainer/               # console app: generate -> train -> evaluate -> save
│   └── ChileMining.DesktopApp/            # WPF app: interactive grade & blast assistant
├── tests/
│   └── ChileMining.Core.Tests/            # xUnit: 7 tests
├── data/                                  # CSVs + trained .zip models (generated, gitignored)
├── README.md
└── README.es.md
```

Open `ChileMining.sln` directly in Visual Studio -- solution, project references, and NuGet packages (`Microsoft.ML`, `Microsoft.ML.FastTree`) are all wired up and ready to build with F5.

## 4. The two ML tasks

**Grade estimation (regression, FastTree)** -- predicts copper grade (%) from `ProfundidadM`, `UnidadGeologica`, `TipoAlteracion`, `DensidadGrCm3`, `ResistividadOhmM`, `DistanciaFallaM`. The synthetic generator ties grade to geology the way real porphyry-copper deposits zone: potassic-altered porphyry/skarn cores get the highest base grade, propylitic-altered andesite the lowest, with density and resistivity correlated to grade through plausible physical relationships (more sulfides → denser rock, lower resistivity) -- not independent random noise.

**Blast fragmentation (multiclass, SDCA)** -- predicts `Fino` / `Medio` / `Grueso` / `SobreTamano` from `BurdenM`, `EspaciamientoM`, `FactorPotenciaKgTon`, `DurezaRocaMpa`, `DiametroPerforacionMm`, using a Kuz-Ram-style fragmentation index (higher powder factor and tighter blast pattern → finer fragmentation; harder rock → coarser) to assign labels before adding noise -- so, like the grade model, the label is causally grounded in the features rather than sampled independently of them.

## 5. Setup

Requires the [.NET 8 SDK](https://dotnet.microsoft.com/download) (or Visual Studio 2022+ with the ".NET desktop development" workload, which includes it).

```powershell
git clone https://github.com/Rxyxs/chile-mining-grade-blast-quality-dotnet.git
cd chile-mining-grade-blast-quality-dotnet
dotnet restore
```

## 6. Usage

**1. Generate data, train, and evaluate both models:**

```powershell
dotnet run --project src/ChileMining.Trainer
```

Writes `drill_holes.csv`, `blast_designs.csv`, `grade_estimator.zip`, and `fragmentation_classifier.zip` to `data/`.

**2. Launch the desktop app** (after step 1 has run at least once):

```powershell
dotnet run --project src/ChileMining.DesktopApp
```

Two tabs: **Control de Leyes** (enter drill-hole parameters, get an estimated grade + ore/waste classification against an illustrative cutoff) and **Diseño de Tronadura** (enter blast design parameters, get a predicted fragmentation category).

**3. Run the tests:**

```powershell
dotnet test
```

Or open `ChileMining.sln` in Visual Studio and use Test Explorer / F5 directly.

## 7. Validated results

All numbers below come from actually running `ChileMining.Trainer` in this repo:

| Metric | Value |
|---|---|
| Drill-hole samples generated | 2,000 |
| Blast design samples generated | 2,000 |
| Grade estimator -- R² | **0.833** |
| Grade estimator -- RMSE | 0.121 (grade units, i.e. ±0.12 pp of Cu%) |
| Grade estimator -- MAE | 0.096 |
| Fragmentation classifier -- MicroAccuracy | **0.823** (vs. ~0.25 random-chance baseline for 4 classes) |
| Fragmentation classifier -- MacroAccuracy | 0.798 |
| Fragmentation classifier -- LogLoss | 0.431 |
| xUnit tests | **7/7 passing** |

Two of the xUnit tests specifically guard against a classic ML bug class: a classifier whose label isn't actually correlated with its features looks fine until you check the metrics and find near-random performance. Here, `PotasicaAlteration_HasHigherAverageGrade_ThanPropilitica` and `HigherPowderFactor_ProducesFinerFragmentation_OnAverage` assert the causal relationship in the generator directly, and `TrainAndEvaluate_LearnsRealSignal_*` assert the trained models clear a real-signal threshold, not just "the code runs."

## 8. A culture-formatting bug worth knowing about

`SyntheticDataGenerator`'s CSV writers use `FormattableString.Invariant(...)` for every numeric field. Without it, on a machine set to a Spanish (Chile) locale, `$"{value}"` formats floats with a **comma** decimal separator (`50,5`) -- which corrupts the CSV, since comma is also the column delimiter. This is guarded by a dedicated regression test (`SaveDrillHolesToCsv_UsesInvariantCulture_RegardlessOfSystemCulture`) that temporarily switches `CurrentCulture` to `es-CL` and asserts the file still parses as 7 columns per row. The WPF app takes the opposite, deliberate approach for *user-facing text*: it parses input with `CurrentCulture` first (so a Chilean user can type `0,30` naturally) and falls back to `InvariantCulture` (so the dot-formatted XAML defaults still work) -- file I/O wants portability, UI text wants to match what the user actually typed.

## 9. Disclaimer

All data is synthetic, generated by `SyntheticDataGenerator` with a fixed seed. The cutoff grade used in the desktop app (0.30% Cu) is illustrative, not a real economic cutoff -- a real cutoff depends on mine/plant costs and the copper price, and isn't a fixed constant.

## License

MIT -- see [LICENSE](LICENSE).

## Author

**Pablo Reyes** -- [github.com/Rxyxs](https://github.com/Rxyxs)
