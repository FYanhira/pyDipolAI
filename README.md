# ⚡ pyDipolAI

A Python-based tool for fitting dielectric spectroscopy data using classic and fractional models.  
Includes Debye, Cole-Cole, Havriliak-Negami, and fractional models, with plans to integrate fuzzy logic and Bayesian optimization.  

---

## ✨ Features
- 🖥️ GUI-based model fitting for dielectric spectroscopy.
- 📊 Visual comparison of multiple dielectric models.
- 🧩 Modular design using Python classes for easy extension.
- ⚙️ Adjustable parameters and instant plotting of fitted curves.

---

## 📈 Models Implemented
- Debye
- Cole-Cole
- Cole-Davidson
- Havriliak-Negami
- Fractional cap-resistor models *(1CR, 2CR)*

---

## 🚀 FBN Project
- 🤖 Integrate fuzzy logic for intelligent model selection.
- 📉 Add Bayesian optimization for parameter fitting.
- 🧪 Include datasets for more polymers and experimental cases.

---

## 📂 Project Structure
pyDipolAI/
│
├── models/               # Model classes (Debye, Cole–Cole, etc.)
├── DielectricModelUI.py  # Main GUI application
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── .gitignore

## 🖼️ Screenshots

### 1. Main GUI Window
![GUI Main Window](assets/gui_main.png)

### 2. Load Dataset
![Load Dataset](assets/load_dataset.png)

### 3. Example of Model Fitting
![Model Fit Example](assets/model_fit.png)

## 🚀 Usage Example

Here’s how to fit a dielectric model using `pyDipolAI`:

# ---------------------------------------------------------
# STEP 1: Import the necessary libraries
# ---------------------------------------------------------
import pandas as pd
import numpy as np
from dielectric_models.models.colecole import ColeColeModel
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# STEP 2: Load your experimental data
# Your CSV file must contain: frequency, eps_real, eps_imag
# ---------------------------------------------------------
data = pd.read_csv("my_experimental_data.csv")
frequency = data['frequency'].values
eps_real = data['eps_real'].values
eps_imag = data['eps_imag'].values

# ---------------------------------------------------------
# STEP 3: Initialize the Cole–Cole model
# ---------------------------------------------------------
model = ColeColeModel()

# ---------------------------------------------------------
# STEP 4: Automatically estimate initial parameters
# This uses your real part data to guess eps_s and eps_inf
# ---------------------------------------------------------
model.set_auto_params_from_data(eps_real)

# ---------------------------------------------------------
# STEP 5: Fit the model to your data
# ---------------------------------------------------------
result_real, result_imag = model.fit(frequency, eps_real, eps_imag)

# ---------------------------------------------------------
# STEP 6: Display the fitted parameters
# ---------------------------------------------------------
print("📌 Fitted Parameters:")
for name, param in model.params.items():
    print(f"{name}: {param.value:.4f} ± {param.stderr:.4e}")

# ---------------------------------------------------------
# STEP 7: Plot the original data and fitted curves
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))

# Real part
plt.plot(frequency, eps_real, 'bo', label="Real Data")
plt.plot(frequency, result_real.best_fit, 'b-', label="Real Fit")

# Imaginary part
plt.plot(frequency, eps_imag, 'ro', label="Imag Data")
plt.plot(frequency, result_imag.best_fit, 'r-', label="Imag Fit")

plt.xscale('log')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Permittivity")
plt.legend()
plt.title("Cole–Cole Model Fit")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.show()


📜 License
MIT License — Free to use, modify, and distribute with attribution.

🤝 Contributing
Contributions are welcome!
Please fork the repository, create a new branch, and submit a pull request.

📧 Contact
Developed by Flor Yanhira Rentería Baltiérrez
🔗 GitHub: FYanhira

