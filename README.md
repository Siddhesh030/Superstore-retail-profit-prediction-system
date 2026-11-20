# 🏪 Superstore Profit & Margin Prediction System | NexusProfit Enterprise

<p align="center">
  <a href="https://superstore-prediction.vercel.app/">
    <img src="https://img.shields.io/badge/Vercel-Live%20Demo-000000?style=for-the-badge&logo=vercel" alt="Live Demo">
  </a>
  <a href="https://github.com/Siddhesh030/Superstore-prediction">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Framework-Flask%203.x-000000?style=for-the-badge&logo=flask" alt="Flask 3.x">
  <img src="https://img.shields.io/badge/ML%20Model-Gradient%20Boosting%20Regressor-F7931E?style=for-the-badge&logo=scikit-learn" alt="Scikit-Learn GBR">
</p>

An end-to-end Machine Learning web application and business intelligence platform designed to forecast order-level profitability and profit margins for retail operations using the **Superstore** dataset.

Powered by a trained **100-tree Gradient Boosting Regressor (GBR)** and served via a high-performance **Flask** microservice on **Vercel Serverless**, featuring interactive sensitivity curves, distribution charts, preset retail scenarios, and instant margin vector telemetry.

---

## 🌐 Live Application

### 👉 [Launch NexusProfit Enterprise Live Application](https://superstore-prediction.vercel.app/)

Simulate retail orders, test discount sensitivity, and analyze profit yields live in your browser.

---

## 📌 Table of Contents

- [Executive Summary](#-executive-summary)
- [Key Features](#-key-features)
- [Machine Learning Architecture](#-machine-learning-architecture)
- [Feature Space & Schema](#-feature-space--schema)
- [Interactive Studio Deck](#-interactive-studio-deck)
- [REST API Reference](#-rest-api-reference)
- [Technology Stack](#-technology-stack)
- [Project Directory Structure](#-project-directory-structure)
- [Local Installation & Setup](#-local-installation--setup)
- [Deployment on Vercel](#-deployment-on-vercel)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 💼 Executive Summary

In commercial retail and e-commerce, maximizing top-line sales without tracking bottom-line profit yields often leads to margin dilution. Factors such as aggressive discount rates, product sub-categories, shipping methods, and geographic distribution directly impact profitability.

**NexusProfit Enterprise** solves this problem by providing real-time predictive insights before transactions are processed:
1. **Accretive vs. Dilutive Analysis**: Instantly detects whether a discount or product mix will generate profit or incur operational loss.
2. **Breakeven Floor Estimation**: Calculates the minimum absorption floor to protect margins.
3. **Discount Curve Sensitivity**: Dynamically plots simulated discount rates from 0% to 70% against projected profits.

---

## ✨ Key Features

* **🤖 Trained Gradient Boosting Regressor**: Employs an ensemble of 100 decision trees (`Gradient_model.pkl`) to capture non-linear interactions between pricing, discounts, shipping speed, and catalog sub-categories.
* **⚡ One-Click Retail Presets**: Includes preconfigured industry scenarios for fast simulation:
  * *Standard Office Bulk*: Moderate sales with 10% discount.
  * *Technology Premium*: High-value phones/copiers with 0% discount resulting in strong profit yields.
  * *Loss-Leader Clearance*: Deep 70% discount illustrating margin degradation.
* **📊 Dual Dynamic Visualizations**:
  * *Sensitivity Analysis*: Real-time Chart.js spline curve simulating profit across varying discount levels.
  * *Revenue & Yield Doughnut Chart*: Visualizes cost base absorption vs. retained net profit.
* **🛡️ Resilient Cross-Version Model Bridge**: Custom unpickler logic that automatically bridges Cython module structures across Scikit-learn versions for seamless cloud hosting.
* **🎨 Modern Responsive UI**: Styled with Tailwind CSS, Alpine.js, and custom dark enterprise aesthetics.

---

## 🧠 Machine Learning Architecture

The predictive core is built using a supervised **Gradient Boosting Regressor**:

```text
  Raw Retail Input Vector (13 Variables)
                   │
                   ▼
  Feature Ordering & Standardization
                   │
                   ▼
  ┌──────────────────────────────────────────────┐
  │  Gradient Boosting Regressor (100 Estimators) │
  │  - Base Estimator: DecisionTreeRegressor     │
  │  - Criterion: Friedman MSE                   │
  │  - Learning Rate: 0.10                       │
  │  - Max Depth: 3                              │
  │  - Loss: Squared Error                       │
  └──────────────────────────────────────────────┘
                   │
                   ▼
  Predicted Order Profit ($) & Margin Ratio (%)
```

### Model Hyperparameters:
* **Estimator Count (`n_estimators`)**: 100 trees
* **Learning Rate (`learning_rate`)**: 0.10
* **Max Depth (`max_depth`)**: 3
* **Min Samples Split (`min_samples_split`)**: 2
* **Loss Function**: `squared_error` (`HalfSquaredError`)

---

## 📋 Feature Space & Schema

The model evaluates **13 discrete and continuous attributes**:

| Category | Feature Name | Description | Range / Values |
|---|---|---|---|
| **Financial** | `Sales` | Gross transaction value | `$1.00` – `$10,000.00` |
| **Financial** | `Discount` | Discount rate applied | `0.00` (0%) – `0.80` (80%) |
| **Financial** | `Quantity` | Number of units ordered | `1` – `50` units |
| **Logistics** | `Ship Mode` | Shipping service tier | `0`: First Class, `1`: Same Day, `2`: Second Class, `3`: Standard Class |
| **Catalog** | `Category` | High-level department | `0`: Furniture, `1`: Office Supplies, `2`: Technology |
| **Catalog** | `Sub-Category` | Specific item classification | 17 categories (Accessories, Binders, Chairs, Phones, etc.) |
| **Customer** | `Segment` | Purchasing customer type | `0`: Consumer, `1`: Corporate, `2`: Home Office |
| **Logistics** | `Region` | Geographic territory | `0`: Central, `1`: East, `2`: South, `3`: West |
| **Logistics** | `State` | Normalized state code | Index `0` – `48` |
| **Logistics** | `City` | Normalized metropolitan area | Index `0` – `500` |
| **Customer** | `Customer Name`| Account identifier | Index `0` – `792` |
| **Catalog** | `Product Name` | SKU identifier | Index `0` – `1849` |
| **Logistics** | `Country` | Country origin | `0`: United States |

---

## 🔌 REST API Reference

The application exposes a lightweight JSON API for programmatic margin scoring.

### `POST /predict`

#### Request Headers:
```http
Content-Type: application/json
```

#### Request Payload:
```json
{
  "features": {
    "Sales": 250.0,
    "Discount": 0.10,
    "Quantity": 3,
    "Ship Mode": 0,
    "Category": 0,
    "Sub-Category": 3,
    "Segment": 0,
    "Region": 1,
    "State": 10,
    "City": 20,
    "Customer Name": 10,
    "Product Name": 100,
    "Country": 0
  }
}
```

#### Successful Response (`200 OK`):
```json
{
  "status": "success",
  "prediction": 42.58,
  "model_type": "GradientBoostingRegressor"
}
```

---

## 🛠️ Technology Stack

* **Backend Engine**: [Flask 3.x](https://flask.palletsprojects.com/) (Python 3.10+)
* **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/), NumPy, Pandas
* **Frontend UI**: [Tailwind CSS](https://tailwindcss.com/) (CDN), [Alpine.js](https://alpinejs.dev/), [Chart.js](https://www.chartjs.org/)
* **Typography**: Plus Jakarta Sans, JetBrains Mono
* **Deployment & Hosting**: [Vercel](https://vercel.com/) Serverless Functions (`@vercel/python`)

---

## 📁 Project Directory Structure

```text
Superstore-prediction/
├── app.py                 # Flask server, ML inference pipeline & Alpine.js template
├── Gradient_model.pkl     # Trained GradientBoostingRegressor scikit-learn model
├── requirements.txt       # Production dependencies (Flask, scikit-learn, numpy, pandas)
├── vercel.json            # Serverless deployment configuration for Vercel
└── README.md              # Documentation
```

---

## 💻 Local Installation & Setup

To run the application locally on your workstation:

### 1. Clone the Repository
```bash
git clone https://github.com/Siddhesh030/Superstore-prediction.git
cd Superstore-prediction
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Launch the Server
```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your web browser.

---

## ☁️ Deployment on Vercel

The project is natively configured for zero-configuration deployments on **Vercel**:

1. Fork or push this repository to your GitHub account.
2. Sign in to your [Vercel Dashboard](https://vercel.com/).
3. Click **Add New Project** and select `Superstore-prediction`.
4. Leave root settings as default (`vercel.json` automatically manages routing).
5. Click **Deploy**.

Any new push to the `main` branch automatically triggers a fresh deployment.

---

## 👨‍💻 Author

**Siddhesh**
* **GitHub**: [@Siddhesh030](https://github.com/Siddhesh030)
* **Project Repository**: [Superstore-prediction](https://github.com/Siddhesh030/Superstore-prediction)
* **Live Deployment**: [https://superstore-prediction.vercel.app](https://superstore-prediction.vercel.app)

---

## ⭐ Support

If this predictive analytics and retail margin optimization project is helpful to your work or learning, please consider starring ⭐ the repository!
