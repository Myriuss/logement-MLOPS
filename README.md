# Projet MLOps — Prédiction du prix d'un logement

Ce projet déploie un modèle de Machine Learning de régression linéaire dans une application complète composée de :

- une API FastAPI
- une interface utilisateur Streamlit
- une conteneurisation Docker
- une pipeline GitHub Actions

## Structure

```text
.
├── .github/workflows/ci-cd.yml
├── api/
├── data/
├── model/
├── ui/
├── docker-compose.yml
└── requirements.txt
```

## Lancer le projet en local

### 1. Entraîner / régénérer le modèle

```bash
python model/train.py
```

### 2. Lancer avec Docker Compose

```bash
docker compose up --build
```

## Accès

- API FastAPI : `http://localhost:8000`
- Documentation Swagger : `http://localhost:8000/docs`
- Interface Streamlit : `http://localhost:8501`

## Commandes Docker utiles

### Build manuel

```bash
docker build -t mlops-logement-api -f api/Dockerfile .
docker build -t mlops-logement-ui -f ui/Dockerfile .
```

### Exécution manuelle

```bash
docker run -p 8000:8000 mlops-logement-api
```

```bash
docker run -e API_URL=http://host.docker.internal:8000/predict -p 8501:8501 mlops-logement-ui
```

## Pipeline CI/CD

La pipeline GitHub Actions :

1. installe les dépendances
2. entraîne le modèle
3. vérifie que l'API charge bien le modèle
4. build l'image Docker de l'API
5. build l'image Docker de l'UI

La partie déploiement AWS sera ajoutée ensuite avec :

- Amazon ECR pour stocker les images Docker
- EC2, ECS ou App Runner pour exécuter l'application

## Variables utiles

- `API_URL` : URL de l'endpoint `/predict` utilisée par Streamlit

## Exemple de payload

```json
{
  "surface": 75,
  "pieces": 3,
  "distance_centre": 5,
  "etage": 2,
  "annee_construction": 2015
}
```
