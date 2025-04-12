# ------------------------------------------------------------
# VNA Data Plotter - L(f), C(f), R(f), X(f) with SRF & Q Markers
# Author: DeXTeR (YO3HEX / YP3X)
# Date: 12 / 04 / 2025
# ------------------------------------------------------------
# This software is provided "as-is" without any warranties or guarantees.
# Use it at your own risk. The author is not responsible for any damage
# or loss resulting from the use or misuse of this script.
# 
# You are free to modify and use this script for personal or educational
# purposes. If redistributed, please retain this header.
# 
# License: MIT License — A permissive license that allows reuse, modification,
# and distribution with attribution and inclusion of the license text.
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

plt.ion()

def parse_siglent_csv(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.strip().startswith("Trace Data"):
            data_start_idx = i + 2
            break
    else:
        raise ValueError("Trace Data section not found.")

    freq, R, X = [], [], []

    for line in lines[data_start_idx:]:
        parts = line.strip().split(',')
        if len(parts) < 3:
            continue
        try:
            freq.append(float(parts[0]))
            R.append(float(parts[1]))
            X.append(float(parts[2]))
        except ValueError:
            continue

    freq = np.array(freq)
    R = np.array(R)
    X = np.array(X)

    L = np.where(X > 0, X / (2 * np.pi * freq), np.nan)
    C = np.where(X < 0, -1 / (2 * np.pi * freq * X), np.nan)

    return freq, R, L, C, X

def parse_s1p_file(filepath):
    freq = []
    mag_db = []
    phase_deg = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line or line.startswith('!'):
                continue
            parts = list(map(float, line.split()))
            freq.append(parts[0])
            mag_db.append(parts[1])
            phase_deg.append(parts[2])

    freq = np.array(freq)
    mag = 10 ** (np.array(mag_db) / 20)
    phase_rad = np.deg2rad(phase_deg)
    s11 = mag * np.exp(1j * phase_rad)

    Z0 = 50
    Z = Z0 * (1 + s11) / (1 - s11)
    R = np.real(Z)
    X = np.imag(Z)

    L = np.where(X > 0, X / (2 * np.pi * freq), np.nan)
    C = np.where(X < 0, -1 / (2 * np.pi * freq * X), np.nan)

    return freq, R, L, C, X

def find_srf(freq, X):
    sign_change = np.diff(np.sign(X))
    zero_crossings = np.where(sign_change != 0)[0]
    if len(zero_crossings) == 0:
        return None
    idx = zero_crossings[0]
    x1, x2 = X[idx], X[idx + 1]
    f1, f2 = freq[idx], freq[idx + 1]
    if x2 - x1 == 0:
        return f1
    return f1 - x1 * (f2 - f1) / (x2 - x1)

def add_hover_annotation(fig, ax, data_series, ylabel, unit, yscale=1.0):
    annot = ax.annotate("", xy=(0, 0), xytext=(15, 15),
                        textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(line, x, y):
        annot.xy = (x, y)
        text = f"Freq: {x:.2f} MHz\n{ylabel}: {y:.2f} {unit}"
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor('lightyellow')
        annot.get_bbox_patch().set_alpha(0.9)

    def hover(event):
        if not event.inaxes:
            annot.set_visible(False)
            fig.canvas.draw_idle()
            return

        for line in data_series:
            cont, ind = line.contains(event)
            if cont:
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                idx = ind["ind"][0]
                x, y = xdata[idx], ydata[idx]
                update_annot(line, x, y)
                annot.set_visible(True)
                fig.canvas.draw_idle()
                return

        annot.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

def plot_trace(fig, ax, title, xlabel, ylabel, unit, data, yscale=1.0):
    fig.canvas.manager.set_window_title(title)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    return []

def main():
    root = tk.Tk()
    root.withdraw()

    filepaths = filedialog.askopenfilenames(
        title="Select Siglent CSV and/or S1P files",
        filetypes=[("Data Files", "*.csv *.s1p")]
    )
    if not filepaths:
        print("No files selected. Exiting.")
        return

    all_data = []

    for filepath in filepaths:
        try:
            ext = os.path.splitext(filepath)[-1].lower()
            if ext == '.csv':
                f, R, L, C, X = parse_siglent_csv(filepath)
            elif ext == '.s1p':
                f, R, L, C, X = parse_s1p_file(filepath)
            else:
                print(f"Unsupported file type: {filepath}")
                continue

            label = os.path.basename(filepath)
            all_data.append((f, R, L, C, X, label))
        except Exception as e:
            print(f"Error parsing file {filepath}: {e}")
            continue

    if not all_data:
        print("No data to plot.")
        return

    titles = [
        ("Inductance L(f)", "L [µH]", "L", "µH", 1e6),
        ("Capacitance C(f)", "C [pF]", "C", "pF", 1e12),
        ("Resistance R(f)", "R [Ω]", "R", "Ω", 1),
        ("Reactance X(f)", "X [Ω]", "X", "Ω", 1)
    ]

    for title, ylabel, key, unit, yscale in titles:
        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title(title)
        ax.set_title(title)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel(ylabel)
        ax.grid(True)

        lines = []
        for f, R, L, C, X, label in all_data:
            data = {"L": L, "C": C, "R": R, "X": X}[key]
            if not np.isnan(data).all():
                line, = ax.plot(f / 1e6, data * yscale, label=label)
                lines.append(line)

                srf_freq = find_srf(f, X)
                if srf_freq:
                    color = line.get_color()
                    ax.axvline(srf_freq / 1e6, color=color, linestyle='--', linewidth=1)
                    ax.text(srf_freq / 1e6, ax.get_ylim()[1] * 0.9,
                            f'SRF: {srf_freq/1e6:.2f} MHz', rotation=90,
                            va='top', ha='center', fontsize=8, color=color)

        ax.legend()
        add_hover_annotation(fig, ax, lines, key, unit, yscale=yscale)

    plt.show(block=False)
    plt.pause(0.1)
    input("Press Enter to exit and close all plots...")

if __name__ == "__main__":
    main()
