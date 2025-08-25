import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        """
    <div style="font-family: Arial, serif; ">
          <h1 style="font-family: Arial, serif;">The Bloch-McConnell Equations and Exchange Phenomenon</h1>
          <h2 style="font-family: Arial, serif;"><b>The Theory</b></h2>
          <p>This paragraph uses Arial.</p>
          </div>
    """
    )
    return


@app.cell
def _(mo):
    # left_img_md  = mo.md('<img src="public/exchange-illustration.png" width="600">')
    # right_img_md = mo.md('<img src="pubic/bloch-mcconnell.png"   width="600">')

    # mo.hstack([left_img_md, right_img_md])
    import base64, pathlib

    p = pathlib.Path("public/exchange-illustration.png")  # or "exchange-illustration.png"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    left_img_md = mo.md(f'<img src="data:image/png;base64,{data}" width="500">')

    p = pathlib.Path("public/bloch-mcconnell.png")  # or "exchange-illustration.png"
    data = base64.b64encode(p.read_bytes()).decode("ascii")

    right_img_md = mo.md(f'<img src="data:image/png;base64,{data}" width="500">')

    mo.hstack([left_img_md, right_img_md])
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import altair as alt
    import pandas as pd
    from mpmath import mp, matrix, expm
    from scipy.fft import fft, fftshift

    alt.data_transformers.disable_max_rows()
    alt.renderers.set_embed_options(actions={"export": True, "source": False, "editor": False})

    # Set precision for mpmath
    mp.dps = 50  # Set decimal places for precision
    return alt, expm, fft, fftshift, matrix, mo, np, pd


@app.cell
def _(mo):


    kAB_log = mo.ui.slider(start=-1.0, stop=6.0, step=0.01, value=0.0,  label="", debounce=True)
    get_kbas, set_kbas = mo.state(0) 

    kBA_log = mo.ui.slider(start=-1.0, stop=6.0, step=0.01, value=get_kbas(),  
                           on_change=set_kbas, label="", debounce=True,
                           #disabled=equal_k.value
                          )
    R2A_log = mo.ui.slider(start=-1.0, stop=3.0, step=0.01, value=1.7, label="", debounce=True)
    R2B_log = mo.ui.slider(start=-1.0, stop=3.0, step=0.01, value=1.7, label="", debounce=True)

    # ΔΩ sliders (linear, can be negative)
    deltaA = mo.ui.slider(start=-1000, stop=1000, step=1, value=-400, label="", debounce=True)
    deltaB = mo.ui.slider(start=-1000, stop=1000, step=1, value= 500, label="", debounce=True)

    # spectrum y-limits (like ft_min/ft_max)
    ft_min = mo.ui.slider(start=-20.0, stop=0.0,   step=0.1, value=-2.0, label="", debounce=True)
    ft_max = mo.ui.slider(start=  1.0, stop=200.0, step=0.1, value=50.0, label="", debounce=True)

    fix_k = mo.ui.radio(
        options={"Free k_BA": "free", "Equal k_AB": "equal"},
        value="Free k_BA",
        label=""
    )
    equal_k = mo.ui.checkbox(value=False, label="Set k_BA = k_AB")
    # show controls
    # mo.vstack([
    #     mo.hstack([kAB_log, kBA_log]),
    #     mo.hstack([R2A_log, R2B_log]),
    #     mo.hstack([deltaA,  deltaB]),
    #     mo.hstack([ft_min,  ft_max]),
    # ])
    return (
        R2A_log,
        R2B_log,
        deltaA,
        deltaB,
        equal_k,
        ft_max,
        ft_min,
        kAB_log,
        kBA_log,
        set_kbas,
    )


@app.cell
def _(mo):
    mo.md(f"""##**Controls and Spectra**""")
    return


@app.cell
def _(
    R2A_log,
    R2B_log,
    R2_A,
    R2_B,
    deltaA,
    deltaB,
    equal_k,
    ft_max,
    ft_min,
    kAB_log,
    kBA_log,
    k_AB,
    k_BA,
    mo,
    set_kbas,
):

    if equal_k.value == True:
        set_kbas(kAB_log.value)
    mo.md(
        f""" 

            |  |   |
            |:---|:----:|
            |{kAB_log} $k_{{AB}}$ = **{k_AB:.4g} Hz**  | |
            |{kBA_log} $k_{{BA}}$ = **{k_BA:.4g} Hz**  |\ {equal_k} |
            |{R2A_log} $R_{{2,A}}$ = **{R2_A:.4g} Hz**  | |
            |{R2B_log} $R_{{2,B}}$ = **{R2_B:.4g} Hz**  |**Set Intensity Scale**|
            |{deltaA} $\Delta\Omega_{{A}}$ = **{deltaA.value:.4g} Hz**  | {ft_min} $I_{{min}}$ = **{ft_min.value:.4g} (Arb)**|
            |{deltaB} $\Delta\Omega_{{B}}$ = **{deltaB.value:.4g} Hz**  | {ft_max} $I_{{max}}$ = **{ft_max.value:.4g} (Arb)** | 

    """
    )
    return


@app.cell
def _(fid_chart, spec_chart):
    (spec_chart & fid_chart).resolve_scale(x='independent', y='independent')

    return


@app.cell
def _(expm, fft, fftshift, matrix, np):



    # Define the Bloch-McConnell matrix for two-state exchange
    # def bloch_mcconnell_matrix(k_AB, k_BA, delta_omega_A, delta_omega_B, R2_A, R2_B):
    #     # Chemical shifts in rad/s
    #     omega_A = 2 * np.pi * delta_omega_A
    #     omega_B = 2 * np.pi * delta_omega_B

    #     # Relaxation and exchange matrix
    #     L = matrix([[-R2_A - k_AB, k_BA],
    #                 [k_AB, -R2_B - k_BA]])

    #     # Off-diagonal terms: chemical shift contributions
    #     H = matrix([[omega_A, 0],
    #                 [0, omega_B]])

    #     # Combined relaxation-exchange matrix
    #     M = L + 1j * H

    #     return M


    def bloch_mcconnell_matrix(
        k_AB, k_BA,
        delta_omega_A, delta_omega_B,   # chemical-shift offsets (Hz or rad/s)
        R2_A, R2_B,
        shifts_in_hz=True,              # True if delta_omega_* given in Hz
        sign='plus'                    # 'minus' => -iΔω (common); 'plus' => +iΔω
    ):
        """
        2×2 complex Bloch–McConnell matrix for transverse magnetization F_A, F_B.
        Units: k, R2 in s^-1 (Hz). delta_omega_* in Hz (default) or rad/s.

        d/dt [F_A; F_B] = M @ [F_A; F_B]
        """
        # Convert shifts to angular frequency (rad/s) if given in Hz
        ωA = 2*np.pi*delta_omega_A if shifts_in_hz else float(delta_omega_A)
        ωB = 2*np.pi*delta_omega_B if shifts_in_hz else float(delta_omega_B)

        # Real (decay + exchange) part
        # L = np.array([[-(R2_A + k_AB),      k_BA],
        #               [     k_AB,      -(R2_B + k_BA)]], dtype=np.complex128)

        # Imag (precession) part with chosen sign
        j = 1j if sign == 'plus' else -1j
        H = np.array([[j*ωA,   0.0],
                      [ 0.0,  j*ωB]], dtype=np.complex128)

        L = np.array([[-R2_A,      0],
                      [     0,      -R2_B ]], dtype=np.complex128)

        E = np.array([[-k_AB,      k_BA],
                      [     k_AB,      -k_BA ]], dtype=np.complex128)

        return L + H + E

    # Function to simulate the FID (Free Induction Decay)
    def simulate_fid(k_AB, k_BA, ΔΩ_Α, ΔΩ_B, R2_A, R2_B, t_max=0.5, n_points=1024):
        # Time points for simulation
        time_points = np.linspace(0, t_max, n_points)

        pA = (k_AB + k_BA)/k_BA
        pB = (k_AB + k_BA)/k_AB
        M0 = matrix([1/pA, 1/pB])

        # Simulate the time evolution of magnetization
        fid = np.zeros(n_points, dtype=complex)

        for i, t in enumerate(time_points):
            M = bloch_mcconnell_matrix(k_AB, k_BA, ΔΩ_Α, ΔΩ_B, R2_A, R2_B)
            M_t = expm(M * t) * M0
            fid[i] = M_t[0] + M_t[1]  # Sum contributions from both states

        return time_points, fid

    # Function to compute the spectrum from the FID
    def compute_spectrum(fid, time_points):
        # Perform Fourier Transform to get the spectrum

        spectrum = fftshift(fft(fid))

        # Frequency axis
        freq_axis = np.fft.fftfreq(2*len(time_points), d=(time_points[1] - time_points[0]))
        freq_axis = fftshift(freq_axis)

        return freq_axis, spectrum


    # Time parameters
    t_max = .1      # Maximum time in seconds
    n_points = 256  # Number of points in time domain
    #time_points = np.linspace(0, t_max, n_points)

    # Parameters for the two-state exchange
    #k_AB = 500.0*10   # Rate constant for A -> B (s^-1)
    #k_BA = 500.0*10   # Rate constant for B -> A (s^-1)
    #ΔΩ_Α = -600.0  # Chemical shift for site A (Hz)
    #ΔΩ_B = 400  # Chemical shift for site B (Hz)
    #R2_A = 20.0    # Transverse relaxation rate for site A (s^-1)
    #R2_B = 20.0    # Transverse relaxation rate for site B (s^-1)
    #ft_min = -20
    #ft_max = 100
    return compute_spectrum, n_points, simulate_fid, t_max


@app.cell
def _(
    R2A_log,
    R2B_log,
    alt,
    compute_spectrum,
    deltaA,
    deltaB,
    equal_k,
    ft_max,
    ft_min,
    kAB_log,
    kBA_log,
    n_points,
    np,
    pd,
    simulate_fid,
    t_max,
):


    # ----- parameters (convert from log sliders) -----
    k_AB = 10**kAB_log.value
    #k_BA = 10**kBA_log.value
    k_BA = 10**(kAB_log.value if equal_k.value == True else kBA_log.value)

    R2_A = 10**R2A_log.value
    R2_B = 10**R2B_log.value
    ΔΩ_Α = float(deltaA.value)
    ΔΩ_B = float(deltaB.value)




    # t_max and n_points should come from your earlier cells; if not, set defaults here:
    # t_max = 0.2
    # n_points = 2048

    # ----- simulate FID -----
    time_points, fid = simulate_fid(k_AB, k_BA, ΔΩ_Α, ΔΩ_B, R2_A, R2_B,
                                    t_max=t_max, n_points=n_points)

    # FID chart data
    df_fid = pd.DataFrame({
        "t": time_points,
        "Real": np.real(fid),
        "Imag": np.imag(fid),
    }).melt(id_vars="t", var_name="component", value_name="value")

    # ----- window (cosine), zero-fill, spectrum -----
    w = np.cos((time_points / t_max) * (np.pi / 2.0))
    fid_w = fid * w
    fid_w[0] = fid[0] * 0.5

    zeros = np.zeros_like(fid_w)
    fid_w_zf = np.concatenate([fid_w, zeros])

    freq_axis, spectrum = compute_spectrum(fid_w_zf, time_points)
    df_spec = pd.DataFrame({"f": freq_axis, "I": spectrum.real})

    # ----- probabilities & labels -----
    k_ex = k_AB + k_BA
    p_A  = (k_BA / k_ex) if k_ex > 0 else 0.0
    p_B  = (k_AB / k_ex) if k_ex > 0 else 0.0
    delta_diff_hz   = abs(ΔΩ_Α - ΔΩ_B)
    delta_diff_rads = 2*np.pi*delta_diff_hz

    title_spec = (
        f"NMR Spectrum with Two-State Exchange\n"
        f"|ΔΩ_A - ΔΩ_B| = {delta_diff_hz:.0f} Hz "
        f"(= {delta_diff_rads:.0f} rad/s),  k_ex = {k_ex:.0f} Hz"
    )

    # ----- Altair: FID (top-right) -----
    fid_chart = (
        alt.Chart(df_fid)
          .mark_line()
          .encode(
              x=alt.X("t:Q", title="Time (s)"),
              y=alt.Y("value:Q", title="FID"),
              color=alt.Color("component:N", title=""),
          )
          .properties(width=1000, height=160, title="FID (Real & Imag)")
          .interactive()
    )

    # ----- Altair: Spectrum (top-left) with vertical rules at ΔΩ_A / ΔΩ_B -----
    spec_line = (
        alt.Chart(df_spec)
          .mark_line(color="black")
          .encode(
              x=alt.X("f:Q", title="Frequency (Hz)"),
              y=alt.Y("I:Q",
                      title="Intensity",
                      scale=alt.Scale(domain=[ft_min.value, ft_max.value])),
          )
          .properties(width=1000, height=160, title=title_spec)
    )

    rule_A = (
        alt.Chart(pd.DataFrame({"f": [ΔΩ_Α], "label": [f"ΔΩ_A, p_A={p_A:.2f}"]}))
          .mark_rule(color="red", strokeDash=[4,4], strokeWidth=2)
          .encode(x="f:Q")
    )
    text_A = (
        alt.Chart(pd.DataFrame({"f": [ΔΩ_Α], "y": [ft_max.value*0.9],
                                "txt": [f"ΔΩ_A  (p_A={p_A:.2f})"]}))
          .mark_text(color="red", dy=-5, angle=0)
          .encode(x="f:Q", y=alt.Y("y:Q"), text="txt:N")
    )

    rule_B = (
        alt.Chart(pd.DataFrame({"f": [ΔΩ_B], "label": [f"ΔΩ_B, p_B={p_B:.2f}"]}))
          .mark_rule(color="blue", strokeDash=[4,4], strokeWidth=2)
          .encode(x="f:Q")
    )
    text_B = (
        alt.Chart(pd.DataFrame({"f": [ΔΩ_B], "y": [ft_max.value*0.8],
                                "txt": [f"ΔΩ_B  (p_B={p_B:.2f})"]}))
          .mark_text(color="blue", dy=-5, angle=0)
          .encode(x="f:Q", y=alt.Y("y:Q"), text="txt:N")
    )

    spec_chart = (spec_line + rule_A + rule_B + text_A + text_B).interactive()

    # ----- lay out like your 2×2 grid’s top row -----

    return R2_A, R2_B, fid_chart, k_AB, k_BA, spec_chart


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
