---
title: Banking-Analytics-Dashboard
colorFrom: blue
colorTo: indigo
sdk: docker
---

<div align="center">

<h1>🏦 Banking Analytics Dashboard</h1>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=6366F1&center=true&vCenter=true&width=700&lines=AI-Powered+Loan+Default+Prediction;Customer+Segmentation+%26+Risk+Scoring;Interactive+Banking+Portfolio+Analytics" alt="Typing SVG"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-4f46e5?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-3b82f6?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)](#)

<br/>

[![Live Demo](https://img.shields.io/badge/%E2%96%B6_Live_Demo-Hugging_Face_Space-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/mnoorchenar/Banking-Analytics-Dashboard)
[![Source Code](https://img.shields.io/badge/Source_Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar/Banking-Analytics-Dashboard)

<br/>

**🏦 Banking Analytics Dashboard** — An interactive, AI-powered Flask dashboard that explores synthetic banking portfolio data through loan default prediction, customer risk segmentation, geographic analysis, and ML model explainability — all served from a single containerised application.

<br/>

---

</div>

## Table of Contents

- [Features](#-features)
- [Architecture](#️-architecture)
- [Getting Started](#-getting-started)
- [Docker Deployment](#-docker-deployment)
- [Dashboard Modules](#-dashboard-modules)
- [ML Models](#-ml-models)
- [Project Structure](#-project-structure)
- [Author](#-author)
- [Contributing](#-contributing)
- [Disclaimer](#️-disclaimer)
- [License](#-license)

---

## ✨ Features

<table>
  <tr>
    <td>🤖 <b>Default Prediction</b></td>
    <td>Binary classification of loan default risk using Random Forest and Gradient Boosting with ROC-AUC evaluation</td>
  </tr>
  <tr>
    <td>👥 <b>Customer Segmentation</b></td>
    <td>KMeans clustering (k=4) on balance, credit score, income, and loan amount to identify distinct customer risk profiles</td>
  </tr>
  <tr>
    <td>📊 <b>Interactive Visualisations</b></td>
    <td>11 fully interactive Plotly charts across 6 dashboard pages — histograms, scatter plots, ROC curves, confusion matrices, and more</td>
  </tr>
  <tr>
    <td>🗺️ <b>Geographic Insights</b></td>
    <td>Regional breakdown of customer counts and default rates across Canadian provinces with dual-axis charting</td>
  </tr>
  <tr>
    <td>🐳 <b>Containerized Deployment</b></td>
    <td>Docker-first architecture, cloud-ready and scalable</td>
  </tr>
</table>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│             Banking Analytics Dashboard                 │
│                                                         │
│  ┌───────────┐    ┌───────────┐    ┌───────────────┐  │
│  │ Synthetic │───▶│    ML     │───▶│   Flask API   │  │
│  │   Data    │    │  Engine   │    │   Backend     │  │
│  │ Generator │    │  (RF/GB/  │    │  (app.py)     │  │
│  └───────────┘    │  KMeans)  │    └───────┬───────┘  │
│                   └───────────┘            │           │
│                                   ┌────────▼────────┐  │
│                                   │  Plotly + Jinja │  │
│                                   │  HTML Dashboard │  │
│                                   │  (Bootstrap 5)  │  │
│                                   └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Git

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/mnoorchenar/Banking-Analytics-Dashboard.git
cd Banking-Analytics-Dashboard

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py
```

Open your browser at `http://localhost:7860` 🎉

---

## 🐳 Docker Deployment

```bash

docker build -t banking-analytics-dashboard .

docker run -p 7860:7860 banking-analytics-dashboard
```

---

## 📊 Dashboard Modules

| Module | Description | Status |
|--------|-------------|--------|
| 🏠 Overview | 8 KPI cards, monthly default trend, portfolio pie, credit score histogram, income-balance scatter | ✅ Live |
| 📦 Portfolio Analysis | Loan amount distribution by segment, income vs balance scatter, segment share donut | ✅ Live |
| ⚠️ Risk Analysis | Credit score histogram by default status, monthly trend line, loan box-plot by segment | ✅ Live |
| 🗺️ Geography | Regional customer count & default rate dual-axis chart, scatter and distribution breakdowns | ✅ Live |
| 🤖 Prediction Models | ROC curves (RF vs GB), confusion matrices, Random Forest feature importance bar chart | ✅ Live |
| 👥 Segmentation | KMeans cluster scatter (balance × credit score), segment share & income/balance plots | 🔄 Beta |
| 🔍 Data Explorer | Scrollable, formatted table of synthetic records with default status badges | ✅ Live |

---

## 🧠 ML Models

```python
# Core Models Used in Banking Analytics Dashboard
models = {
    "default_classifier_primary"   : "RandomForestClassifier(n_estimators=120, max_depth=8, class_weight='balanced')",
    "default_classifier_secondary" : "GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)",
    "customer_segmentation"        : "KMeans(n_clusters=4, n_init=10)",
    "feature_scaling"              : "StandardScaler() — applied to [balance, credit_score, loan_amount, income]",
    "evaluation_metrics"           : "ROC-AUC, Accuracy, Confusion Matrix, Classification Report"
}
```

---

## 📁 Project Structure

```
Banking-Analytics-Dashboard/
│
├── 📄 Dockerfile                 # Container definition (python:3.11-slim, port 7860)
├── 📄 app.py                     # Application entry point — data generation,
└── 📄 requirements.txt           # Python dependencies
```

---

## 👨‍💻 Author

<div align="center">

<table>
<tr>
<td align="center" width="100%">

<img src="https://avatars.githubusercontent.com/mnoorchenar" width="120" style="border-radius:50%; border: 3px solid #4f46e5;" alt="Mohammad Noorchenarboo"/>

<h3>Mohammad Noorchenarboo</h3>

<code>Data Scientist</code> &nbsp;|&nbsp; <code>AI Researcher</code> &nbsp;|&nbsp; <code>Biostatistician</code>

📍 &nbsp;Ontario, Canada &nbsp;&nbsp; 📧 &nbsp;[mohammadnoorchenarboo@gmail.com](mailto:mohammadnoorchenarboo@gmail.com)

──────────────────────────────────────

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mnoorchenar)&nbsp;
[![Personal Site](https://img.shields.io/badge/Website-mnoorchenar.github.io-4f46e5?style=for-the-badge&logo=githubpages&logoColor=white)](https://mnoorchenar.github.io/)&nbsp;
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar)&nbsp;
[![Google Scholar](https://img.shields.io/badge/Scholar-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white)](https://scholar.google.ca/citations?user=nn_Toq0AAAAJ&hl=en)&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar)

</td>
</tr>
</table>

</div>

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

---

## ⚠️ Disclaimer

> [!WARNING]
> This project is developed strictly for educational and research purposes and does not constitute professional advice of any kind. All datasets used are synthetically generated — no real customer, financial, or banking data is stored or processed. This software is provided "as is" without warranty of any kind; use at your own risk.

---

## 📜 License

Distributed under the **MIT License**.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:3b82f6,100:4f46e5&height=120&section=footer&text=Made%20with%20%E2%9D%A4%EF%B8%8F%20by%20Mohammad%20Noorchenarboo&fontColor=ffffff&fontSize=18&fontAlignY=80" width="100%"/>

[![GitHub Stars](https://img.shields.io/github/stars/mnoorchenar/Banking-Analytics-Dashboard?style=social)](https://github.com/mnoorchenar/Banking-Analytics-Dashboard/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/mnoorchenar/Banking-Analytics-Dashboard?style=social)](https://github.com/mnoorchenar/Banking-Analytics-Dashboard/fork)

<sub>The name "Banking Analytics Dashboard" is used purely for academic and research purposes. Any similarity to existing product names or trademarks is entirely coincidental. This project has no affiliation with any commercial entity.</sub>

</div>
