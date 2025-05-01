# ------------------------------------------------------------
# VNA Capacitor Analyzer - C(f), R(f), Q(f), |Z(f)| with SRF Markers
# Author: DeXTeR (YO3HEX / YP3X)
# Date: 29 / 04 / 2025 (Updated)
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
from scipy.signal import savgol_filter

plt.ion()

GOOD_R_THRESHOLD = 1  # Adjustable Resistance Threshold for "good" zone in ohms
MAX_VALID_Q = 1500     # Configurable Q filtering threshold
MIN_VALID_R = 0.001    # Configurable R filtering threshold
NUM_AVG_SCANS = 10  # Set how many scans to average (max 10)

import pandas as pd
import pyvisa


def fetch_data_from_siglent():
    SIGLENT_IP = "192.168.0.35"  # Replace with your actual IP address
    rm = pyvisa.ResourceManager()
    instr = rm.open_resource(f"TCPIP::{SIGLENT_IP}::INSTR")
    instr.timeout = 30000
    instr.write_termination = '\n'
    instr.read_termination = '\n'

    instr.write(":CALC1:PAR1:DEF S11")
    instr.write(":CALC1:PAR1:SEL")
    instr.write(":CALC1:FORM SMITH")
    instr.write(":SENS1:FREQ:STAR 1MHz")
    instr.write(":SENS1:FREQ:STOP 30MHz")
    instr.write(":SENS1:SWE:POIN 750")
    instr.write(":INIT:CONT ON")
    instr.write(":INIT")
    instr.query("*OPC?")

    raw_data = instr.query_ascii_values(":CALC1:DATA:FDAT?", container=np.array)
    s11_re = raw_data[::2]
    s11_im = raw_data[1::2]
    s11_complex = s11_re + 1j * s11_im

    z0 = 50
    zin = z0 * (1 + s11_complex) / (1 - s11_complex)

    f_start = float(instr.query(":SENS1:FREQ:STAR?"))
    f_stop = float(instr.query(":SENS1:FREQ:STOP?"))
    n_points = int(instr.query(":SENS1:SWE:POIN?"))
    freqs = np.linspace(f_start, f_stop, n_points)

    df = pd.DataFrame({
        'Frequency_Hz': freqs,
        'Re_Z': np.real(zin),
        'Im_Z': np.imag(zin)
    })
    return df

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
    if hasattr(fig.canvas, '_hover_callback_id'):
        fig.canvas.mpl_disconnect(fig.canvas._hover_callback_id)

    annot = ax.annotate("", xy=(0, 0), xytext=(15, 15),
                        textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(line, x, y):
        annot.xy = (x, y)
        text = f"""Freq: {x:.2f} MHz\n{ylabel}: {y:.2f} {unit}"""
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

    fig.canvas._hover_callback_id = fig.canvas.mpl_connect("motion_notify_event", hover)

def main():
    all_data = []

    def acquire_and_plot():
        
        df_list = [fetch_data_from_siglent() for _ in range(NUM_AVG_SCANS)]
        df = sum(df_list) / len(df_list)
        freq = df['Frequency_Hz'].to_numpy()
        R = df['Re_Z'].to_numpy()
        X = df['Im_Z'].to_numpy()

        # Interpolate invalid R values (below threshold)
        R_filtered = np.copy(R)
        R_invalid_mask = R < MIN_VALID_R
        if np.any(R_invalid_mask) and len(R) > 2:
            R_filtered[R_invalid_mask] = np.interp(
                freq[R_invalid_mask],
                freq[~R_invalid_mask],
                R[~R_invalid_mask]
            )
        R = R_filtered

        C = np.where(X != 0, 1 / (2 * np.pi * freq * np.abs(X)), np.nan)
        Q = np.where(R != 0, np.abs(X) / R, np.nan)

        # Apply Q filtering: values above threshold or below 0 are invalid and interpolated
        Q_filtered = np.copy(Q)
        Q_invalid_mask = (Q > MAX_VALID_Q) | (Q < 0)
        if np.any(Q_invalid_mask) and len(Q) > 2:
            Q_filtered[Q_invalid_mask] = np.interp(
                freq[Q_invalid_mask],
                freq[~Q_invalid_mask],
                Q[~Q_invalid_mask]
            )
        Q = Q_filtered

        label = f"Scan {len(all_data)+1}"
        all_data.append((freq, R, C, Q, X, label))

        titles = [
            ("Capacitance C(f)", "C [pF]", "C", "pF", 1e12),
            ("Resistance R(f)", "R [Ω]", "R", "Ω", 1),
            ("Quality Factor Q(f)", "Q", "Q", "", 1),
            ("Impedance |Z(f)|", "|Z| [Ω]", "Zmag", "Ω", 1)
        ]

        for title, ylabel, key, unit, yscale in titles:
            fig = plt.figure(num=title)
            fig.clf()
            ax = fig.add_subplot(111)
            ax.set_title(title)
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel(ylabel)
            ax.grid(True)

            lines = []
            for f, R, C, Q, X, label in all_data:
                raw_data = {"C": C, "R": R, "Q": Q, "Zmag": np.sqrt(R**2 + X**2)}[key]
                if not np.isnan(raw_data).all():
                    srf_freq = find_srf(f, X)
                    new_label = label
                    if srf_freq:
                        new_label += f" (SRF: {srf_freq/1e6:.2f} MHz)"

                    if len(raw_data) > 7:
                        data = savgol_filter(raw_data, window_length=21, polyorder=3, mode='mirror')
                    else:
                        data = raw_data

                    line, = ax.plot(f / 1e6, data * yscale, label=new_label)
                    lines.append(line)

                    if srf_freq:
                        color = line.get_color()
                        ax.axvline(srf_freq / 1e6, color=color, linestyle='--', linewidth=1)

            ax.legend()
            add_hover_annotation(fig, ax, lines, key, unit, yscale=yscale)
        plt.draw()

    # GUI button
    root = tk.Tk()
    root.title("VNA Capacitor Analyzer")
    tk.Button(root, text="New Scan + Overlay", command=acquire_and_plot).pack(padx=20, pady=20)
    tk.Button(root, text="Quit", command=root.quit).pack(padx=20, pady=(0, 20))

    acquire_and_plot()
    plt.pause(0.1)
    root.mainloop()
    plt.close('all')

if __name__ == "__main__":
    main()
