import streamlit as st
from fractions import Fraction
import pandas as pd

# ---------------------------
# Utilidades numéricas
# ---------------------------
def F(x):
    if isinstance(x, Fraction):
        return x
    s = str(x).strip()
    if s == "":
        return Fraction(0)
    return Fraction(s)

def fmt(fr: Fraction) -> str:
    if fr.denominator == 1:
        return str(fr.numerator)
    return f"{fr.numerator}/{fr.denominator}"

def safe_int(x, default=2):
    try:
        return int(x)
    except:
        return default

# ---------------------------
# Simplex (MAX) con logs
# ---------------------------
def tableau_to_df(tableau, basis, var_names):
    rows = []
    for i, row in enumerate(tableau):
        base = var_names[basis[i]] if i < len(basis) else "Obj(z-c)"
        rows.append([base] + [fmt(v) for v in row])
    cols = ["Base"] + var_names + ["RHS"]
    return pd.DataFrame(rows, columns=cols)

def pivot(tableau, basis, r, c):
    piv = tableau[r][c]
    if piv == 0:
        raise ZeroDivisionError("Pivote 0")
    tableau[r] = [v / piv for v in tableau[r]]
    for i in range(len(tableau)):
        if i == r:
            continue
        factor = tableau[i][c]
        if factor != 0:
            tableau[i] = [
                tableau[i][j] - factor * tableau[r][j]
                for j in range(len(tableau[i]))
            ]
    basis[r] = c

def build_objective_row(tableau, basis, c):
    nvars = len(tableau[0]) - 1
    obj = [F(0)] * (nvars + 1)
    # fila tipo (z - c)
    for j in range(nvars):
        obj[j] = -c[j]
    obj[-1] = F(0)

    for i, bv in enumerate(basis):
        cb = c[bv]
        if cb != 0:
            for j in range(nvars + 1):
                obj[j] += cb * tableau[i][j]
    return obj

def choose_entering(tableau):
    obj = tableau[-1]
    candidates = [j for j in range(len(obj) - 1) if obj[j] < 0]
    if not candidates:
        return None
    # Bland: el menor índice evita ciclos
    return min(candidates)

def choose_leaving(tableau, basis, enter_col):
    ratios = []
    for i in range(len(tableau) - 1):
        a = tableau[i][enter_col]
        if a > 0:
            ratios.append((tableau[i][-1] / a, i))
    if not ratios:
        return None
    min_ratio = min(ratios, key=lambda x: x[0])[0]
    tied = [i for (ratio, i) in ratios if ratio == min_ratio]
    # Bland: desempate por índice de la básica
    tied.sort(key=lambda i: basis[i])
    return tied[0]

def simplex_collect(tableau, basis, var_names, c, title_prefix=""):
    tableau = [row[:] for row in tableau]
    basis = basis[:]
    logs = []

    obj = build_objective_row(tableau, basis, c)
    tableau.append(obj)
    logs.append((f"{title_prefix}Tableau inicial", tableau_to_df(tableau, basis, var_names)))

    it = 0
    while True:
        enter = choose_entering(tableau)
        if enter is None:
            logs.append((f"{title_prefix}Óptimo alcanzado ✅", None))
            break

        leave = choose_leaving(tableau, basis, enter)
        if leave is None:
            raise ValueError("Problema NO acotado (unbounded): no hay fila saliente válida.")

        it += 1
        msg = (
            f"{title_prefix}Iteración {it}: "
            f"Entra **{var_names[enter]}** | "
            f"Sale **{var_names[basis[leave]]}** | "
            f"Pivote (fila {leave+1}, col {enter+1})"
        )
        pivot(tableau, basis, leave, enter)
        logs.append((msg, tableau_to_df(tableau, basis, var_names)))

    return tableau, basis, logs

# ---------------------------
# Pasar a forma estándar (soporta <=, >=, =)
# ---------------------------
def standardize(constraints, n_vars):
    var_names = [f"x{i+1}" for i in range(n_vars)]
    tableau = []
    basis = []
    artificial_idx = []

    def add_var(name):
        var_names.append(name)
        for r in tableau:
            r.insert(-1, F(0))
        return len(var_names) - 1

    for k, cons in enumerate(constraints, start=1):
        a = [F(v) for v in cons["a"]]
        b = F(cons["b"])
        sense = cons["sense"].strip()

        # Si RHS negativo: multiplicar por -1 y voltear desigualdad
        if b < 0:
            a = [-v for v in a]
            b = -b
            if sense == "<=":
                sense = ">="
            elif sense == ">=":
                sense = "<="

        row = a[:]
        row += [F(0)] * (len(var_names) - len(row))
        row.append(b)
        tableau.append(row)

        if sense == "<=":
            s = add_var(f"s{k}")
            tableau[-1][s] = F(1)
            basis.append(s)

        elif sense == ">=":
            e = add_var(f"e{k}")
            tableau[-1][e] = F(-1)
            aidx = add_var(f"a{k}")
            tableau[-1][aidx] = F(1)
            basis.append(aidx)
            artificial_idx.append(aidx)

        elif sense == "=":
            aidx = add_var(f"a{k}")
            tableau[-1][aidx] = F(1)
            basis.append(aidx)
            artificial_idx.append(aidx)
        else:
            raise ValueError(f"Signo inválido: {sense}")

        # Igualar longitudes
        for i in range(len(tableau)):
            need = len(var_names) + 1 - len(tableau[i])
            if need > 0:
                tableau[i] = tableau[i][:-1] + [F(0)] * need + [tableau[i][-1]]

    return tableau, basis, var_names, artificial_idx

def drop_artificials(tableau, basis, var_names, artificial_idx):
    art_set = set(artificial_idx)

    # sacar artificiales de la base si aparecen
    i = 0
    while i < len(basis):
        bv = basis[i]
        if bv in art_set:
            pivot_col = None
            for j in range(len(var_names)):
                if j not in art_set and j != bv and tableau[i][j] != 0:
                    pivot_col = j
                    break
            if pivot_col is not None:
                dummy_obj = [F(0)] * (len(var_names) + 1)
                full = [r[:] for r in tableau] + [dummy_obj]
                pivot(full, basis, i, pivot_col)
                tableau = full[:-1]
                i += 1
            else:
                if tableau[i][-1] == 0:
                    tableau.pop(i)
                    basis.pop(i)
                else:
                    raise ValueError("Inviable: artificial básica con RHS != 0.")
        else:
            i += 1

    # eliminar columnas artificiales
    for col in sorted(list(art_set), reverse=True):
        for r in tableau:
            r.pop(col)
        var_names.pop(col)
        for k in range(len(basis)):
            if basis[k] > col:
                basis[k] -= 1

    return tableau, basis, var_names

def extract_solution(tableau, basis, var_names, n_original):
    sol = {name: Fraction(0) for name in var_names[:n_original]}
    for i, bv in enumerate(basis):
        if bv < n_original:
            sol[var_names[bv]] = tableau[i][-1]
    return sol

# ---------------------------
# UI bonita
# ---------------------------
st.set_page_config(page_title="Simplex paso a paso", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .title {font-size: 2.2rem; font-weight: 800; margin-bottom: 0.25rem;}
    .subtitle {opacity: .8; margin-top: 0;}
    .card {
        padding: 1rem 1.2rem;
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        background: rgba(255,255,255,0.03);
        margin-bottom: 1rem;
    }
    .small {opacity: .8; font-size: 0.95rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📊 Simplex (2 fases) paso a paso</div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Metes el modelo, y te devuelve Fase I y Fase II con tablas, pivotes y la solución óptima.</p>', unsafe_allow_html=True)

# Estado inicial
if "n_vars" not in st.session_state:
    st.session_state.n_vars = 2
if "constraints" not in st.session_state:
    st.session_state.constraints = [
        {"a": ["3", "1"], "sense": "=",  "b": "3"},
        {"a": ["4", "3"], "sense": ">=", "b": "6"},
        {"a": ["1", "2"], "sense": "<=", "b": "4"},
    ]
if "c" not in st.session_state:
    st.session_state.c = ["4", "1"]
if "obj_sense" not in st.session_state:
    st.session_state.obj_sense = "Minimizar"

with st.sidebar:
    st.header("⚙️ Configuración")

    st.session_state.obj_sense = st.selectbox(
        "Objetivo",
        ["Minimizar", "Maximizar"],
        index=0 if st.session_state.obj_sense == "Minimizar" else 1
    )

    st.session_state.n_vars = st.slider("Número de variables (x1, x2, ...)", 2, 6, st.session_state.n_vars)

    st.markdown("**Coeficientes del objetivo**")
    c_inputs = []
    for i in range(st.session_state.n_vars):
        default = st.session_state.c[i] if i < len(st.session_state.c) else "0"
        c_inputs.append(st.text_input(f"Coef de x{i+1}", value=default, key=f"c_{i}"))
    st.session_state.c = c_inputs

    st.divider()
    st.markdown("**Restricciones**")
    st.caption("Puedes escribir enteros, decimales o fracciones tipo 3/5.")

    colb1, colb2 = st.columns(2)
    if colb1.button("➕ Agregar restricción"):
        st.session_state.constraints.append(
            {"a": ["0"] * st.session_state.n_vars, "sense": "<=", "b": "0"}
        )
    if colb2.button("➖ Quitar última"):
        if len(st.session_state.constraints) > 1:
            st.session_state.constraints.pop()

    # Render restricciones
    new_constraints = []
    for idx, cons in enumerate(st.session_state.constraints, start=1):
        st.markdown(f"**R{idx}**")
        row_cols = st.columns([1,1,1,1,1,1,1])
        a_vals = []
        for j in range(st.session_state.n_vars):
            a_vals.append(row_cols[j].text_input(f"a{idx}_{j}", value=cons["a"][j] if j < len(cons["a"]) else "0", label_visibility="collapsed"))
        sense = row_cols[st.session_state.n_vars].selectbox(
            f"sense_{idx}", ["<=", ">=", "="],
            index=["<=", ">=", "="].index(cons["sense"]),
            label_visibility="collapsed"
        )
        b = row_cols[st.session_state.n_vars+1].text_input(f"b_{idx}", value=cons["b"], label_visibility="collapsed")
        new_constraints.append({"a": a_vals, "sense": sense, "b": b})
    st.session_state.constraints = new_constraints

    st.divider()
    solve = st.button("🚀 Resolver y mostrar paso a paso", use_container_width=True)

# Mostrar el modelo en pantalla
def latex_model(obj_sense, c, constraints):
    terms = []
    for i, ci in enumerate(c, start=1):
        ciF = F(ci)
        if ciF == 0:
            continue
        terms.append(f"{fmt(ciF)}x_{i}")
    if not terms:
        terms = ["0"]

    obj = " \\min " if obj_sense == "Minimizar" else " \\max "
    s = obj + "Z = " + " + ".join(terms) + "\\\\"
    s += "\\text{s.a. }\\\\"
    for cons in constraints:
        lhs = []
        for i, ai in enumerate(cons["a"], start=1):
            aiF = F(ai)
            if aiF == 0:
                continue
            lhs.append(f"{fmt(aiF)}x_{i}")
        if not lhs:
            lhs = ["0"]
        s += " + ".join(lhs) + f"\\ {cons['sense']}\\ {fmt(F(cons['b']))}\\\\"
    s += "x_i \\ge 0"
    return s

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🧾 Tu modelo")
st.latex(latex_model(st.session_state.obj_sense, st.session_state.c, st.session_state.constraints))
st.markdown('<p class="small">Tip: si pones fracciones (ej: 3/5) te saldrá exacto, sin redondeos.</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

tabs = st.tabs(["✅ Resultado", "📌 Fase I", "📌 Fase II", "🆘 Ayuda rápida"])

if solve:
    try:
        n = st.session_state.n_vars
        c = [F(x) for x in st.session_state.c]
        constraints = [{"a": [F(v) for v in cons["a"]], "sense": cons["sense"], "b": F(cons["b"])} for cons in st.session_state.constraints]

        # Estándar + Fase I
        tableau, basis, var_names, artificial_idx = standardize(constraints, n)

        c_phase1 = [F(0)] * len(var_names)
        for idx in artificial_idx:
            c_phase1[idx] = F(-1)  # MAX -W

        t1, b1, logs1 = simplex_collect(tableau, basis, var_names, c_phase1, title_prefix="FASE I: ")

        if t1[-1][-1] != 0:
            with tabs[0]:
                st.error("No hay solución factible (Fase I terminó con W ≠ 0).")
            st.stop()

        tableau1 = t1[:-1]
        basis1 = b1[:]

        # Quitar artificiales
        tableau2, basis2, var_names2 = drop_artificials(tableau1, basis1, var_names, artificial_idx)

        # Fase II
        c_phase2 = [F(0)] * len(var_names2)
        sign = F(-1) if st.session_state.obj_sense == "Minimizar" else F(1)
        for i in range(n):
            c_phase2[i] = sign * c[i]  # MAX (Z) o MAX(-Z)

        t2, b2, logs2 = simplex_collect(tableau2, basis2, var_names2, c_phase2, title_prefix="FASE II: ")

        # Solución
        sol = extract_solution(t2[:-1], b2, var_names2, n)
        # valor óptimo de la función maximizada
        opt_Q = t2[-1][-1]
        if st.session_state.obj_sense == "Minimizar":
            opt_Z = -opt_Q
        else:
            opt_Z = opt_Q

        with tabs[0]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("✅ Solución óptima")
            cols = st.columns(min(4, n))
            for i in range(n):
                cols[i % len(cols)].metric(f"x{i+1}", fmt(sol.get(f"x{i+1}", Fraction(0))))
            st.metric("Valor óptimo Z", fmt(opt_Z))
            st.markdown("</div>", unsafe_allow_html=True)

            # Comprobación rápida
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🔎 Comprobación rápida de restricciones")
            for k, cons in enumerate(constraints, start=1):
                lhs = sum(cons["a"][i] * sol.get(f"x{i+1}", Fraction(0)) for i in range(n))
                b = cons["b"]
                sense = cons["sense"]
                ok = (lhs <= b) if sense == "<=" else (lhs >= b) if sense == ">=" else (lhs == b)
                st.write(f"R{k}: {fmt(lhs)} {sense} {fmt(b)}  →  {'✅' if ok else '❌'}")
            st.markdown("</div>", unsafe_allow_html=True)

            # Descargar reporte
            report_lines = []
            report_lines.append("SIMPLEX 2 FASES - REPORTE\n")
            report_lines.append(f"Objetivo: {st.session_state.obj_sense}\n")
            report_lines.append(f"Z = " + " + ".join([f"{fmt(c[i])}x{i+1}" for i in range(n)]) + "\n\n")
            report_lines.append("Restricciones:\n")
            for k, cons in enumerate(constraints, start=1):
                report_lines.append(
                    "  " + " + ".join([f"{fmt(cons['a'][i])}x{i+1}" for i in range(n)]) +
                    f" {cons['sense']} {fmt(cons['b'])}\n"
                )
            report_lines.append("\nSolución:\n")
            for i in range(n):
                report_lines.append(f"  x{i+1} = {fmt(sol.get(f'x{i+1}', Fraction(0)))}\n")
            report_lines.append(f"  Z* = {fmt(opt_Z)}\n")

            st.download_button(
                "⬇️ Descargar reporte (TXT)",
                data="".join(report_lines),
                file_name="reporte_simplex.txt",
                mime="text/plain"
            )

        with tabs[1]:
            st.subheader("📌 Fase I (paso a paso)")
            for msg, df in logs1:
                with st.expander(msg, expanded=False):
                    if df is not None:
                        st.dataframe(df, use_container_width=True)

        with tabs[2]:
            st.subheader("📌 Fase II (paso a paso)")
            for msg, df in logs2:
                with st.expander(msg, expanded=False):
                    if df is not None:
                        st.dataframe(df, use_container_width=True)

        with tabs[3]:
            st.subheader("🆘 Si algo falla, normalmente es por esto")
            st.write("1) Algún número mal escrito (ej: `3//5` en vez de `3/5`).")
            st.write("2) El problema es inviable (no existe solución que cumpla todas).")
            st.write("3) El problema es no acotado (se puede mejorar infinito).")

    except Exception as e:
        with tabs[0]:
            st.error(f"Error: {e}")
else:
    with tabs[3]:
        st.subheader("🧠 Cómo usarlo (en 20 segundos)")
        st.write("1) En la barra izquierda eliges Minimizar/Maximizar y cuántas variables.")
        st.write("2) Rellenas el objetivo y las restricciones.")
        st.write("3) Le das a **Resolver** y te salen las tablas en Fase I y Fase II.")
