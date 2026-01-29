import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Derivative → Function Reconstructor", layout="wide")

st.title("📈 Derivative → Original Function Reconstructor")

st.markdown("""
Enter a **first derivative** or **second derivative**, supply initial conditions,  
and this app will **reconstruct the original function** and visualize how everything connects.
""")

# Symbol
x = sp.symbols('x')

# --- Sidebar Inputs ---
st.sidebar.header("Inputs")

derivative_type = st.sidebar.radio(
    "What are you providing?",
    ["First Derivative f'(x)", "Second Derivative f''(x)"]
)

derivative_input = st.sidebar.text_input(
    "Enter the derivative (SymPy syntax)",
    value="2*x"
)

x0 = st.sidebar.number_input("Initial x-value (x₀)", value=0.0)
y0 = st.sidebar.number_input("f(x₀) value", value=0.0)

if derivative_type == "Second Derivative f''(x)":
    v0 = st.sidebar.number_input("f'(x₀) value", value=0.0)
else:
    v0 = None

xmin, xmax = st.sidebar.slider("x-range", -10.0, 10.0, (-5.0, 5.0))

# --- Symbolic Math ---
try:
    derivative_expr = sp.sympify(derivative_input)

    if derivative_type == "First Derivative f'(x)":
        C = sp.symbols('C')
        f_expr = sp.integrate(derivative_expr, x) + C
        C_val = sp.solve(f_expr.subs(x, x0) - y0, C)[0]
        f_expr = f_expr.subs(C, C_val)

        fprime_expr = derivative_expr
        fsecond_expr = sp.diff(fprime_expr, x)

    else:
        C1, C2 = sp.symbols('C1 C2')
        fprime_expr = sp.integrate(derivative_expr, x) + C1
        f_expr = sp.integrate(fprime_expr, x) + C2

        solutions = sp.solve([
            f_expr.subs(x, x0) - y0,
            fprime_expr.subs(x, x0) - v0
        ], (C1, C2))

        fprime_expr = fprime_expr.subs(solutions)
        f_expr = f_expr.subs(solutions)
        fsecond_expr = derivative_expr

    # Convert to numerical functions
    f = sp.lambdify(x, f_expr, "numpy")
    fprime = sp.lambdify(x, fprime_expr, "numpy")
    fsecond = sp.lambdify(x, fsecond_expr, "numpy")

    xs = np.linspace(xmin, xmax, 800)

    # --- Critical & Inflection Points ---
    crit_points = sp.solve(sp.Eq(fprime_expr, 0), x)
    infl_points = sp.solve(sp.Eq(fsecond_expr, 0), x)

    crit_points = [float(cp) for cp in crit_points if cp.is_real]
    infl_points = [float(ip) for ip in infl_points if ip.is_real]

    # --- Plotting ---
    fig, axes = plt.subplots(
        3, 1, figsize=(10, 10), sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1]}
    )

    # f(x)
    axes[0].plot(xs, f(xs))
    axes[0].set_ylabel("f(x)")
    axes[0].set_title("Original Function")

    # f'(x)
    axes[1].plot(xs, fprime(xs))
    axes[1].axhline(0, linestyle="--")
    axes[1].set_ylabel("f'(x)")

    # f''(x)
    axes[2].plot(xs, fsecond(xs))
    axes[2].axhline(0, linestyle="--")
    axes[2].set_ylabel("f''(x)")
    axes[2].set_xlabel("x")

    # --- Connector Lines ---
    for cp in crit_points:
        if xmin <= cp <= xmax:
            axes[0].plot([cp, cp], [f(cp), fprime(cp)], linestyle="--")
            axes[1].scatter(cp, 0)

    for ip in infl_points:
        if xmin <= ip <= xmax:
            axes[0].plot([ip, ip], [f(ip), fsecond(ip)], linestyle="--")
            axes[2].scatter(ip, 0)

    st.pyplot(fig)

    st.success("Function successfully reconstructed!")

    st.latex(r"f(x) = " + sp.latex(f_expr))
    st.latex(r"f'(x) = " + sp.latex(fprime_expr))
    st.latex(r"f''(x) = " + sp.latex(fsecond_expr))

except Exception as e:
    st.error("There was an error parsing or solving the function.")
    st.code(str(e))
