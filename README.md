# VNA SRF Plotter

A lightweight but powerful Python utility for analyzing and comparing S11 data from ferrite cores, inductors, or other one-port RF devices. This tool plots **L(f)**, **C(f)**, **R(f)**, and **X(f)** from `.s1p` and Siglent `.csv` files, automatically detects the **Self-Resonant Frequency (SRF)**, and displays the **Q factor** at SRF — all in an intuitive, interactive GUI.

---

## 🔍 Features

- Load multiple `.s1p` and/or Siglent `.csv` files simultaneously
- Interactive plots of:
  - Inductance **L(f)** (µH)
  - Capacitance **C(f)** (pF)
  - Resistance **R(f)** (Ω)
  - Reactance **X(f)** (Ω)
- Automatic **SRF (Self-Resonant Frequency)** detection per trace
- Hover tooltips to inspect values at any frequency point
- Color-coded SRF markers matching the trace
- Clean GUI — no command-line interaction needed

---

## 📁 Supported File Formats

- `.s1p` Touchstone files with:  
  `# Hz S DB R 50`  
  (S11 magnitude in dB, angle in degrees)

- Siglent `.csv` files exported from the SVA1032X or similar, containing:
  ```
  Frequency, R, X, ...
  ```

---

## 🚀 Getting Started

1. Install required packages:
   ```bash
   pip install numpy matplotlib
   ```

2. Run the script:
   ```bash
   python S1P_CSV_IPlotter.py
   ```

3. Select one or more `.s1p` or `.csv` files via the file dialog.

4. Explore the plots and hover for details. Press `Enter` in the terminal to close all plots.

---


## ⚠️ Disclaimer

This software is provided **as-is** without warranty of any kind. Use it at your own risk.  
See the top of the script for full legal and license information.

---

## 📜 License

This project is licensed under the **MIT License** — free to use, modify, and share.

---

## 🧑‍💻 Author

Created by **DeXTeR (YO3HEX / YP3X)**  

---

## 💬 Feedback & Contributions

Ideas, feedback, and improvements are welcome!  
Feel free to open issues, submit pull requests, or just drop a message.
