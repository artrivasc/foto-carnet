# Foto Carnet

App web para generar fotos tipo carnet/pasaporte usando IA.
Soporta hasta 15 fotos en simultáneo con procesamiento en paralelo.

## Estructura

```
foto-carnet/
├── app.py              # Backend Flask
├── requirements.txt    # Dependencias Python
├── render.yaml         # Config para Render.com
└── templates/
    └── index.html      # Frontend completo
```

## Cómo subir a Render.com

1. Crea una cuenta gratis en https://render.com

2. Sube esta carpeta a GitHub:
   - Crea cuenta en https://github.com si no tienes
   - Crea un repositorio nuevo (puede ser privado)
   - Sube los archivos de esta carpeta

3. En Render:
   - New → Web Service
   - Conecta tu repositorio de GitHub
   - Render detecta el render.yaml automáticamente
   - En "Environment Variables" agrega:
       Key:   ANTHROPIC_API_KEY
       Value: tu-clave-de-anthropic
   - Click en "Deploy"

4. En unos 2 minutos tendrás una URL pública lista para compartir.

## API Key de Anthropic

Obtén tu clave en: https://console.anthropic.com
Crea cuenta → API Keys → Create Key

El costo es muy bajo: menos de $0.01 por foto procesada.
