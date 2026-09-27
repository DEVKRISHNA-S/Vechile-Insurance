# 🚗 Vehicle Insurance — End-to-End MLOps Project

> **An end-to-end Machine Learning project focused on production-grade MLOps, automation, cloud deployment, and CI/CD.**

This project implements a **Vehicle Insurance Prediction system**, but the primary objective is not the complexity of the machine learning model.

The project was built to demonstrate how a machine learning system can be taken beyond a Jupyter notebook and transformed into a **structured, reproducible, deployable, and automated ML application**.

The project covers the complete ML lifecycle:

**Data → Validation → Transformation → Training → Evaluation → Model Registry → Deployment → Docker → CI/CD → AWS**

---

## 🎯 Project Objective

The machine learning problem in this project is intentionally simple: predicting outcomes related to vehicle insurance.

The main focus is the **engineering infrastructure around the ML model**.

The project demonstrates how to:

* Structure an ML project for production
* Build modular ML pipeline components
* Store and retrieve data from MongoDB
* Validate incoming datasets against a defined schema
* Perform feature transformation using reusable estimators
* Train and evaluate models through pipeline components
* Store trained models in AWS S3
* Build a prediction pipeline
* Containerize the application using Docker
* Deploy the application on AWS EC2
* Store Docker images using Amazon ECR
* Automate deployment using GitHub Actions
* Use a self-hosted GitHub Actions runner
* Manage secrets and cloud credentials securely through environment variables and GitHub Secrets

---

# 🏗️ MLOps Architecture

```text
                         ┌──────────────────┐
                         │     Developer    │
                         │      GitHub      │
                         └────────┬─────────┘
                                  │
                                  │ git push
                                  ▼
                       ┌──────────────────────┐
                       │   GitHub Actions     │
                       │      CI/CD           │
                       └──────────┬───────────┘
                                  │
                                  │
                    ┌─────────────▼─────────────┐
                    │     Self-Hosted Runner    │
                    │        AWS EC2            │
                    └─────────────┬─────────────┘
                                  │
                            Docker Build
                                  │
                                  ▼
                         ┌────────────────┐
                         │   Amazon ECR   │
                         │ Docker Registry│
                         └───────┬────────┘
                                 │
                                 │ Pull Image
                                 ▼
                         ┌────────────────┐
                         │    AWS EC2     │
                         │ Docker Runtime │
                         └───────┬────────┘
                                 │
                                 ▼
                         ┌────────────────┐
                         │  Prediction    │
                         │     API/App    │
                         └────────────────┘


     ┌────────────────┐
     │  MongoDB Atlas │
     │  Data Storage  │
     └───────┬────────┘
             │
             ▼
     ┌────────────────┐
     │ Data Ingestion │
     └───────┬────────┘
             │
             ▼
     ┌────────────────┐
     │ Data Validation │
     └───────┬────────┘
             │
             ▼
     ┌─────────────────┐
     │ Data             │
     │ Transformation   │
     └───────┬─────────┘
             │
             ▼
     ┌────────────────┐
     │ Model Training │
     └───────┬────────┘
             │
             ▼
     ┌────────────────┐
     │ Model Evaluation│
     └───────┬────────┘
             │
             ▼
     ┌────────────────┐
     │ Model Registry │
     │    AWS S3      │
     └────────────────┘
```

---

# 🧰 Technology Stack

| Area               | Technology                        |
| ------------------ | --------------------------------- |
| Programming        | Python                            |
| Environment        | Conda / Virtual Environment       |
| Data Storage       | MongoDB Atlas                     |
| Data Processing    | Pandas, NumPy                     |
| Machine Learning   | Scikit-learn                      |
| Data Validation    | Custom schema-based validation    |
| Experimentation    | Jupyter Notebook                  |
| Logging            | Python Logging                    |
| Exception Handling | Custom Exception Module           |
| Cloud              | AWS                               |
| Model Storage      | Amazon S3                         |
| Containerization   | Docker                            |
| Container Registry | Amazon ECR                        |
| Compute            | Amazon EC2                        |
| CI/CD              | GitHub Actions                    |
| CI/CD Runner       | Self-hosted GitHub Actions Runner |
| Web Application    | Flask                             |
| Version Control    | Git / GitHub                      |

---

# 🧠 What Makes This an MLOps Project?

The ML model itself is only one component.

The important part of this project is the **system surrounding the model**.

Instead of having:

```text
Notebook → Train Model → Done
```

the project implements:

```text
Data
  ↓
Data Ingestion
  ↓
Data Validation
  ↓
Data Transformation
  ↓
Model Training
  ↓
Model Evaluation
  ↓
Model Registry
  ↓
Prediction Pipeline
  ↓
Docker
  ↓
CI/CD
  ↓
AWS Deployment
```

This separation makes the system easier to:

* maintain
* test
* debug
* reproduce
* deploy
* extend
* automate

---

# 📁 Project Structure

```text
vehicle-insurance-mlops/
│
├── .github/
│   └── workflows/
│       └── aws.yaml
│
├── notebook/
│   ├── mongoDB_demo.ipynb
│   └── EDA_and_Feature_Engineering.ipynb
│
├── src/
│   └── ...
│
├── static/
│
├── templates/
│
├── artifact/
│
├── app.py
├── demo.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── setup.py
├── pyproject.toml
├── template.py
└── README.md
```

The source code is organized into independent components rather than placing the entire ML workflow inside one notebook or script.

---

# 🔄 ML Pipeline

## 1. Data Ingestion

Data is stored in **MongoDB Atlas**.

The ingestion component:

1. Connects to MongoDB
2. Retrieves documents
3. Converts the key-value data into a Pandas DataFrame
4. Stores the resulting artifact for downstream pipeline components

```text
MongoDB Atlas
      ↓
MongoDB Connection
      ↓
Data Access Layer
      ↓
DataFrame
      ↓
Data Ingestion Artifact
```

### Key MLOps concept

**Separation of data access from ML logic.**

The model pipeline does not directly contain database-specific logic.

---

# 2. Data Validation

Before training, incoming data is checked against a predefined schema.

The schema describes information such as:

* expected columns
* data types
* numerical/categorical features
* target information
* dataset structure

This prevents unexpected data from silently entering the training pipeline.

```text
Raw Data
   ↓
Schema Validation
   ↓
Valid Dataset
   │
   └── Invalid → Pipeline Failure
```

### MLOps concept

**Data contracts / schema-based validation**

---

# 3. Data Transformation

The transformation component prepares the validated data for machine learning.

Typical operations include:

* numerical feature transformation
* categorical feature encoding
* missing-value handling
* feature preprocessing
* creation of a reusable transformation pipeline

The preprocessing estimator is persisted so that the **same transformation logic can be reused during inference**.

This is important because training-time and inference-time preprocessing must remain consistent.

```text
Validated Data
      ↓
Preprocessing Pipeline
      ↓
Transformed Features
      ↓
Model Trainer
```

---

# 4. Model Training

The model trainer receives transformed training data and trains the ML model.

The training logic is isolated inside a dedicated pipeline component rather than being coupled to the notebook.

The trained model is then serialized as an artifact.

```text
Training Data
      ↓
Model Trainer
      ↓
Trained Model
      ↓
Model Artifact
```

---

# 5. Model Evaluation

The trained model is evaluated against the existing model.

A configurable threshold is used:

```python
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02
```

This creates a basic mechanism for deciding whether a newly trained model represents a meaningful change before it is pushed to the model registry.

### MLOps concept

**Model validation before promotion**

Rather than automatically deploying every newly trained model:

```text
New Model
   ↓
Evaluation
   ↓
Compare Performance
   ↓
Promotion Decision
```

---

# 6. Model Registry / Model Storage

AWS S3 is used as the remote storage layer for trained models.

```text
Model Evaluation
       ↓
Model Pusher
       ↓
AWS S3
       ↓
model-registry/
```

The project contains an S3 estimator responsible for model operations such as:

* pushing models
* pulling models
* managing model artifacts

This creates a separation between the ML pipeline and the cloud storage implementation.

---

# ☁️ AWS Infrastructure

The deployment infrastructure uses multiple AWS services.

### Amazon S3

Used for:

* model artifact storage
* model registry

### Amazon ECR

Used as a private Docker image registry.

```text
Docker Image
     ↓
Amazon ECR
```

### Amazon EC2

Used as the compute environment for running the deployed application.

```text
EC2
 └── Docker
      └── Vehicle Insurance Application
```

---

# 🐳 Docker Containerization

The application is containerized using Docker.

The Docker image contains the application and its runtime dependencies, allowing the application to run consistently across environments.

```text
Source Code
    ↓
Dockerfile
    ↓
Docker Image
    ↓
Amazon ECR
    ↓
EC2
    ↓
Running Container
```

This addresses the classic:

> "It works on my machine."

problem by packaging the application environment together with the application.

---

# 🔁 CI/CD Pipeline

One of the major goals of the project is automating deployment.

GitHub Actions is used to create a CI/CD workflow.

The workflow is defined in:

```text
.github/
└── workflows/
    └── aws.yaml
```

The deployment flow is approximately:

```text
Developer
   │
   │ git push
   ▼
GitHub Repository
   │
   ▼
GitHub Actions
   │
   ├── Build Docker Image
   │
   ├── Authenticate with AWS
   │
   ├── Push Image to ECR
   │
   └── Deploy to EC2
   │
   ▼
Running Application
```

The result is a deployment workflow where a new code push can automatically trigger the deployment process.

---

# 🖥️ Self-Hosted GitHub Actions Runner

Instead of relying exclusively on GitHub-hosted runners, this project uses an **AWS EC2 instance as a self-hosted GitHub Actions runner**.

```text
GitHub
   │
   │ Workflow
   ▼
EC2 Self-Hosted Runner
   │
   ├── Docker
   ├── AWS CLI
   └── Deployment Commands
```

The runner is registered with the GitHub repository and executes workflow jobs directly on the EC2 machine.

This demonstrates an important CI/CD concept:

**The CI/CD execution environment can be controlled by the organization rather than being entirely managed by the CI provider.**

---

# 🔐 Secrets & Configuration Management

Sensitive credentials are not hard-coded into the source code.

The project uses environment variables and GitHub repository secrets for sensitive configuration.

Examples include:

```text
MONGODB_URL
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
ECR_REPO
```

GitHub Secrets are used by the CI/CD workflow for cloud authentication.

This prevents credentials from being committed directly to Git.

> **Production note:** For a real production environment, IAM roles, workload identity, or other short-lived credential mechanisms should generally be preferred over long-lived access keys.

---

# 📊 Experimentation & EDA

Jupyter notebooks are used during the exploratory phase for:

* exploratory data analysis
* understanding the dataset
* feature engineering
* MongoDB experimentation
* validating assumptions before implementing pipeline components

The notebooks are intentionally separated from the production pipeline.

```text
Notebook
   ↓
Experimentation / EDA
   ↓
Validated Logic
   ↓
Production Pipeline Component
```

This keeps experimentation flexible while keeping production code modular.

---

# 📝 Logging & Exception Handling

The project includes custom modules for:

### Logging

Centralized logging is implemented to make pipeline execution easier to monitor and debug.

### Exception Handling

Custom exception handling provides more meaningful error information when pipeline components fail.

Instead of:

```text
Something went wrong.
```

the pipeline can provide contextual information about:

* which component failed
* where the failure occurred
* what operation caused the failure

These capabilities become increasingly important as ML pipelines become more complex.

---

# 🧩 Modular Pipeline Architecture

The project follows a component-based architecture.

Major pipeline components include:

```text
Data Ingestion
       ↓
Data Validation
       ↓
Data Transformation
       ↓
Model Trainer
       ↓
Model Evaluation
       ↓
Model Pusher
```

Each component has its own:

* configuration
* artifact
* implementation
* responsibilities

This makes it possible to modify one stage without rewriting the entire pipeline.

---

# ⚙️ Configuration & Artifacts

The project separates:

### Configuration

Information required to execute pipeline components.

For example:

```text
DataIngestionConfig
DataValidationConfig
DataTransformationConfig
ModelTrainerConfig
ModelEvaluationConfig
ModelPusherConfig
```

### Artifacts

Outputs generated by each pipeline component.

For example:

```text
DataIngestionArtifact
DataValidationArtifact
DataTransformationArtifact
ModelTrainerArtifact
ModelEvaluationArtifact
ModelPusherArtifact
```

This creates an explicit contract between pipeline stages.

```text
Component A
    │
    │ Artifact
    ▼
Component B
    │
    │ Artifact
    ▼
Component C
```

---

# 🌐 Prediction Pipeline

After model training and evaluation, the trained model is exposed through a prediction application.

The project contains:

```text
app.py
static/
templates/
```

The prediction workflow is:

```text
User Input
    ↓
Web Application
    ↓
Prediction Pipeline
    ↓
Load Model
    ↓
Apply Preprocessing
    ↓
Generate Prediction
    ↓
Return Result
```

The same preprocessing logic used during training is reused during inference.

---

# 🚀 Local Setup

## 1. Clone the Repository

```bash
git clone <repository-url>
cd vehicle-insurance-mlops
```

## 2. Create Environment

```bash
conda create -n vehicle python=3.10 -y
conda activate vehicle
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Install the Local Package

The project uses `setup.py` and `pyproject.toml` to make the source package importable.

```bash
pip list
```

Verify that the project package is available in the environment.

---

# 🗄️ MongoDB Configuration

Create a MongoDB Atlas cluster and obtain the MongoDB connection string.

Set:

```text
MONGODB_URL
```

### PowerShell

```powershell
$env:MONGODB_URL="mongodb+srv://<username>:<password>@..."
```

### Bash

```bash
export MONGODB_URL="mongodb+srv://<username>:<password>@..."
```

The application uses this connection to access the training data stored in MongoDB Atlas.

---

# ☁️ AWS Configuration

Configure the required AWS credentials in the environment.

### PowerShell

```powershell
$env:AWS_ACCESS_KEY_ID="..."
$env:AWS_SECRET_ACCESS_KEY="..."
$env:AWS_DEFAULT_REGION="us-east-1"
```

### Bash

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
```

The project uses:

```text
AWS S3 → Model Storage
AWS ECR → Container Registry
AWS EC2 → Compute / Deployment
```

---

# 🐳 Running with Docker

Build the Docker image:

```bash
docker build -t vehicleproj .
```

Run the container:

```bash
docker run -p 5080:5080 vehicleproj
```

The application can then be accessed through:

```text
http://localhost:5080
```

---

# 🔄 CI/CD Setup

The CI/CD workflow is located at:

```text
.github/workflows/aws.yaml
```

The repository requires the following GitHub Secrets:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
ECR_REPO
```

A typical deployment flow is:

```text
git push
   ↓
GitHub Actions
   ↓
Docker Build
   ↓
Amazon ECR
   ↓
EC2
   ↓
Docker Container
   ↓
Application
```

---

# 🏭 Production Deployment

The application is deployed to an Ubuntu-based AWS EC2 instance.

The EC2 machine provides:

* Docker runtime
* Self-hosted GitHub Actions runner
* Application execution environment

The Docker image is stored in Amazon ECR and used for deployment.

The application exposes port:

```text
5080
```

After configuring the EC2 security group to allow the required inbound traffic, the application can be accessed through:

```text
http://<EC2-PUBLIC-IP>:5080
```

---

# 🔬 MLOps Concepts Demonstrated

This project demonstrates the following concepts:

| MLOps Area             | Implementation                         |
| ---------------------- | -------------------------------------- |
| Project Structure      | Modular Python package                 |
| Environment Management | Conda                                  |
| Dependency Management  | `requirements.txt`                     |
| Package Management     | `setup.py`, `pyproject.toml`           |
| Data Storage           | MongoDB Atlas                          |
| Data Ingestion         | Dedicated ingestion component          |
| Data Validation        | Schema-based validation                |
| Data Transformation    | Reusable preprocessing pipeline        |
| Model Training         | Dedicated trainer component            |
| Model Evaluation       | Threshold-based evaluation             |
| Model Registry         | AWS S3                                 |
| Model Deployment       | AWS EC2                                |
| Containerization       | Docker                                 |
| Container Registry     | Amazon ECR                             |
| CI/CD                  | GitHub Actions                         |
| CI/CD Infrastructure   | Self-hosted runner                     |
| Configuration          | Centralized configuration classes      |
| Artifacts              | Pipeline artifact contracts            |
| Logging                | Custom logging module                  |
| Exception Handling     | Custom exception module                |
| Experimentation        | Jupyter notebooks                      |
| Prediction             | Production prediction pipeline         |
| Secrets                | Environment variables + GitHub Secrets |
| Cloud Infrastructure   | AWS                                    |

---

# 💡 Key Engineering Takeaways

The primary learning outcomes of this project were not related to achieving a highly sophisticated model.

The important engineering lessons were:

### 1. ML code needs software engineering around it

A model is only one part of an ML system.

### 2. Data needs to be treated as a dependency

Data ingestion and validation should be explicit pipeline stages.

### 3. Training and inference must share preprocessing logic

The transformation pipeline should not be recreated manually during prediction.

### 4. Models need lifecycle management

A trained model should be evaluated before being promoted and should be stored in a controlled location.

### 5. Deployment should be reproducible

Docker provides a consistent runtime environment.

### 6. Deployment should be automated

GitHub Actions removes the need to manually execute deployment steps after every change.

### 7. Cloud infrastructure becomes part of the ML system

The ML lifecycle extends beyond Python and includes:

```text
AWS S3
AWS ECR
AWS EC2
GitHub Actions
Docker
MongoDB Atlas
```

---

# 🔮 Possible Future Improvements

The current implementation provides a foundation for further MLOps improvements.

Potential extensions include:

* MLflow for experiment tracking
* MLflow Model Registry
* DVC for dataset versioning
* Automated model retraining
* Automated data drift detection
* Automated model monitoring
* Prometheus/Grafana monitoring
* CloudWatch integration
* Unit and integration testing
* Automated test stages in CI
* Infrastructure as Code using Terraform
* AWS IAM roles instead of long-lived access keys
* ECS/EKS-based deployment
* HTTPS with a reverse proxy
* API-based inference using FastAPI
* Model performance monitoring in production
* Feature store integration
* Canary or blue-green deployment
* Automated rollback on deployment failure

---

# 📌 Project Perspective

The Vehicle Insurance model is intentionally straightforward.

The real purpose of this repository is to demonstrate the transition from:

```text
Machine Learning Experiment
```

to:

```text
Production-Oriented ML System
```

with the following lifecycle:

```text
                 ┌──────────────┐
                 │    Data      │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │  Validation  │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │Transformation│
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │    Training  │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │  Evaluation  │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │ Model Storage│
                 │    AWS S3    │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │   Docker     │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │    ECR       │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │    EC2       │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │   Deployed   │
                 │ Application  │
                 └──────────────┘
```

**The model is the product of the ML pipeline; the MLOps infrastructure is the focus of the project.**
