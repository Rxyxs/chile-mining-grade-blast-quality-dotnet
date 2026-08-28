# Contenedor de dos etapas: compila ChileMining.Trainer y ChileMining.Cli
# (ambas multiplataforma, .NET 8 sin GUI) con el SDK completo, y sirve la
# imagen final desde la imagen runtime-only (sin SDK, ~4x mas liviana).
# ChileMining.DesktopApp (WPF, net8.0-windows) queda deliberadamente fuera
# de esta imagen: WPF no corre en Linux, y no es el caso de uso de un
# contenedor de todos modos.

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

# Copia solo los .csproj primero para que `dotnet restore` se cachee en capas
# de Docker independientes del codigo fuente -- un cambio en un .cs no
# invalida el restore, solo el publish.
COPY src/ChileMining.Core/ChileMining.Core.csproj src/ChileMining.Core/
COPY src/ChileMining.Trainer/ChileMining.Trainer.csproj src/ChileMining.Trainer/
COPY src/ChileMining.Cli/ChileMining.Cli.csproj src/ChileMining.Cli/
RUN dotnet restore src/ChileMining.Trainer/ChileMining.Trainer.csproj \
 && dotnet restore src/ChileMining.Cli/ChileMining.Cli.csproj

COPY src/ChileMining.Core/ src/ChileMining.Core/
COPY src/ChileMining.Trainer/ src/ChileMining.Trainer/
COPY src/ChileMining.Cli/ src/ChileMining.Cli/

RUN dotnet publish src/ChileMining.Trainer/ChileMining.Trainer.csproj -c Release -o /app/trainer --no-self-contained \
 && dotnet publish src/ChileMining.Cli/ChileMining.Cli.csproj -c Release -o /app/cli --no-self-contained

FROM mcr.microsoft.com/dotnet/runtime:8.0 AS runtime
WORKDIR /app
ENV CHILEMINING_DATA_DIR=/app/data
ENV DOTNET_EnableDiagnostics=0

COPY --from=build /app/trainer ./trainer
COPY --from=build /app/cli ./cli

# Entrena y exporta el modelo ONNX una vez, al construir la imagen -- el
# contenedor queda listo para inferir de inmediato con `docker run`, sin un
# paso de entrenamiento adicional que el usuario del contenedor tenga que
# saber ejecutar primero.
RUN dotnet ./trainer/ChileMining.Trainer.dll

ENTRYPOINT ["dotnet", "./cli/chilemining-cli.dll"]
CMD ["--help"]
