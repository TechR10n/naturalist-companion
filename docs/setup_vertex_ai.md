# Vertex AI Setup (Local Dev)

This setup enables the GCP path for `notebooks/anc_gcp.ipynb`.

## Prereqs

- Python 3.12
- `gcloud` CLI installed and authenticated
- A GCP project with billing enabled

## Steps

1. Create and activate a venv

```bash
./scripts/bootstrap_vertex.sh
source .venv/bin/activate
```

2. Configure `.env`

```bash
cp .env.example .env
```

Set at least:
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION`

3. Authenticate (ADC)

```bash
gcloud auth application-default login
```

4. Enable Vertex AI API

```bash
gcloud config set project your-gcp-project-id
gcloud services enable aiplatform.googleapis.com
```

5. Run the notebook

```bash
jupyter lab
```

Open `notebooks/anc_gcp.ipynb` and run all cells.
