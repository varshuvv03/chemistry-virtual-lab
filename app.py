import io
import os
import random
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image as PILImage
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, Lipinski, rdMolDescriptors
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image as RLImage
from reportlab.platypus import (PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
st.set_page_config(page_title="Cheminformatics Virtual Lab", page_icon="🧪")

INSTITUTE = "SRM Institute of Science and Technology, Tiruchirappalli"
DEVELOPERS = [
    ("Varshini", "RA2611030050040"),
    ("Vismitha Parasa", "RA2611030050047"),
]
CONTACTS = ["vismithaparasa14@gmail.com", "varshuvv03@gmail.com"]

EXAMPLES = {
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Phenol": "Oc1ccccc1",
    "Acetic Acid": "CC(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
}
MODELS = ["Ball and Stick", "Stick", "Space Filling", "Wireframe", "Cartoon"]
PAGES = [
    "🏠 Home",
    "📖 Theory",
    "🧬 Molecular Visualization",
    "📊 Molecular Properties Report",
    "📈 Structure–Property Analysis",
    "📝 Assessment",
    "📄 Final Project Report",
]

# Unicode-capable PDF font (DejaVu ships inside matplotlib) so σ, π, Å, ³ print correctly.
FONT, FONT_B = "Helvetica", "Helvetica-Bold"
try:
    _d = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(_d, "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(_d, "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                                  italic="DejaVu", boldItalic="DejaVu-Bold")
    FONT, FONT_B = "DejaVu", "DejaVu-Bold"
except Exception:
    pass


# ------------------------------------------------------------------
# CHEMISTRY HELPERS
# ------------------------------------------------------------------
def get_mol(smiles):
    if not smiles or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())


@st.cache_data(show_spinner=False)
def molblock_3d(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    return Chem.MolToMolBlock(mol)


def draw_2d(mol, size=(600, 400)):
    return Draw.MolToImage(mol, size=size)


def pil_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def calc(mol):
    """All descriptors used in the lab, as raw values."""
    mh = Chem.AddHs(mol)
    kek = Chem.Mol(mh)
    Chem.Kekulize(kek, clearAromaticFlags=True)
    sigma = pi = 0
    for b in kek.GetBonds():
        sigma += 1
        pi += int(round(b.GetBondTypeAsDouble())) - 1
    return {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mw": Descriptors.MolWt(mol),
        "exact": Descriptors.ExactMolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rotb": Lipinski.NumRotatableBonds(mol),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "arom": rdMolDescriptors.CalcNumAromaticRings(mol),
        "aliph": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "atoms": mol.GetNumAtoms(),
        "total_atoms": mh.GetNumAtoms(),
        "heavy": mol.GetNumHeavyAtoms(),
        "hbonds": mol.GetNumBonds(),
        "sigma": sigma,
        "pi": pi,
        "fsp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "mr": Descriptors.MolMR(mol),
    }


def full_table_df(mol, smiles):
    c = calc(mol)
    rows = [
        ("Input SMILES", smiles), ("Canonical SMILES", Chem.MolToSmiles(mol)),
        ("Molecular Formula", c["formula"]), ("Molecular Weight", round(c["mw"], 3)),
        ("Exact Molecular Weight", round(c["exact"], 4)), ("LogP", round(c["logp"], 3)),
        ("TPSA", round(c["tpsa"], 2)), ("Hydrogen Bond Donors", c["hbd"]),
        ("Hydrogen Bond Acceptors", c["hba"]), ("Rotatable Bonds", c["rotb"]),
        ("Ring Count", c["rings"]), ("Aromatic Rings", c["arom"]), ("Aliphatic Rings", c["aliph"]),
        ("Number of Atoms", c["atoms"]), ("Heavy Atoms", c["heavy"]),
        ("Heavy Atom Bonds", c["hbonds"]), ("Sigma (σ) Bonds", c["sigma"]),
        ("Pi (π) Bonds", c["pi"]), ("Total Bonds (σ + π)", c["sigma"] + c["pi"]),
        ("Fraction Csp3", round(c["fsp3"], 3)), ("Molar Refractivity", round(c["mr"], 2)),
    ]
    return pd.DataFrame({"Property": [r[0] for r in rows], "Value": [str(r[1]) for r in rows]})


def interpretation_rows(mol):
    """(Property, Value, Interpretation) rows used in the PDF report."""
    c = calc(mol)
    lp = c["logp"]
    if lp < 0:
        lp_t = "A negative value indicates a preference toward the aqueous phase and relatively high hydrophilicity."
    elif lp < 3:
        lp_t = "A moderate value suggests a balance between hydrophilic and lipophilic character."
    else:
        lp_t = "A high value suggests lipophilic character and lower aqueous solubility."
    tp = c["tpsa"]
    if tp < 40:
        tp_t = "A relatively low polar surface area, indicating limited overall molecular polarity."
    elif tp < 90:
        tp_t = "A moderate polar surface area, indicating a balanced polar character."
    else:
        tp_t = "A high polar surface area, indicating strong polarity and potentially lower membrane permeability."
    if c["rotb"] == 0:
        rb_t = "No freely rotatable bonds are detected, suggesting a relatively rigid molecular framework."
    elif c["rotb"] <= 5:
        rb_t = "A small number of rotatable bonds indicates limited conformational flexibility."
    else:
        rb_t = "Many rotatable bonds indicate a highly flexible molecule."
    fs_t = ("A high sp³ fraction indicates a more saturated, three-dimensional molecular framework."
            if c["fsp3"] >= 0.5 else
            "A lower sp³ fraction indicates a flatter, more unsaturated or aromatic framework.")
    return [
        ("Molecular Formula", c["formula"], "Shows the elemental composition of the molecule and is useful for understanding its basic chemical composition."),
        ("Molecular Weight", f"{c['mw']:.3f}", "Represents the average molecular mass. Molecular size and mass can influence diffusion, formulation and many physicochemical behaviours."),
        ("Exact Molecular Weight", f"{c['exact']:.4f}", "The monoisotopic mass calculated from exact isotopic masses; this is especially useful when comparing mass-spectrometry data."),
        ("LogP", f"{lp:.3f}", lp_t),
        ("TPSA", f"{tp:.2f}", tp_t),
        ("Hydrogen Bond Donors", str(c["hbd"]), f"The molecule has {c['hbd']} donor site(s) capable of donating hydrogen in hydrogen-bond interactions."),
        ("Hydrogen Bond Acceptors", str(c["hba"]), f"The molecule has {c['hba']} acceptor site(s) capable of accepting hydrogen in hydrogen-bond interactions."),
        ("Rotatable Bonds", str(c["rotb"]), rb_t),
        ("Ring Count", str(c["rings"]), f"The molecule contains {c['rings']} ring(s), which contribute to its overall shape and structural rigidity."),
        ("Aromatic Rings", str(c["arom"]), f"{c['arom']} aromatic ring(s) contribute to aromaticity, planarity and delocalized π-electron character."),
        ("Aliphatic Rings", str(c["aliph"]), f"{c['aliph']} non-aromatic ring(s) contribute to cyclic structure without aromatic π-electron delocalization."),
        ("Number of Atoms", str(c["atoms"]), "Counts atoms explicitly present in the RDKit molecular representation; implicit hydrogens are not included here."),
        ("Heavy Atoms", str(c["heavy"]), "Counts all non-hydrogen atoms and provides a useful measure of molecular structural size."),
        ("Heavy Atom Bonds", str(c["hbonds"]), "Counts connections between non-hydrogen atoms in the molecular graph."),
        ("Sigma (σ) Bonds", str(c["sigma"]), "Every bonded atom pair contains one σ bond; this value includes bonds involving explicit or added hydrogens."),
        ("Pi (π) Bonds", str(c["pi"]), "π bonds arise from multiple bonds and aromatic systems and are associated with electron delocalization."),
        ("Total Bonds (σ + π)", str(c["sigma"] + c["pi"]), "Represents the total bond components obtained by adding the calculated σ and π bond counts."),
        ("Fraction Csp3", f"{c['fsp3']:.3g}" if c["fsp3"] not in (0, 1) else f"{c['fsp3']:.1f}", fs_t),
        ("Molar Refractivity", f"{c['mr']:.2f}", "Relates to molecular volume and electronic polarizability and reflects how the electron cloud responds to an electric field."),
    ]


def key_interpretation(mol):
    c = calc(mol)
    lp_word = "more hydrophilic than lipophilic" if c["logp"] < 0 else "more lipophilic than hydrophilic"
    return [
        ("Molecular identity", f"The submitted structure has molecular formula {c['formula']} and molecular weight {c['mw']:.3f}."),
        ("Lipophilicity", f"LogP = {c['logp']:.3f}, indicating that the molecule is {lp_word}. This descriptor describes the balance between preference for non-polar and polar environments."),
        ("Molecular polarity", f"TPSA = {c['tpsa']:.2f} Å². TPSA reflects the contribution of polar atoms and functional groups to the overall surface polarity of the molecule."),
        ("Hydrogen bonding", f"The structure contains {c['hbd']} hydrogen-bond donor(s) and {c['hba']} acceptor(s), which helps describe possible intermolecular interactions."),
        ("Flexibility", f"{c['rotb']} rotatable bond(s) indicate the degree of conformational freedom available to the molecule."),
        ("Cyclic and aromatic character", f"The molecule contains {c['rings']} ring(s), including {c['arom']} aromatic ring(s). Aromatic rings are associated with delocalized π-electron systems."),
        ("Bonding pattern", f"The calculated structure contains {c['sigma']} σ bonds and {c['pi']} π bonds. This helps distinguish the framework-forming σ bonds from multiple-bond and delocalized π bonding."),
        ("Three-dimensional character", f"Fraction Csp³ = {c['fsp3']:.3g}. This descriptor provides a simple indication of the balance between saturated 3D carbon environments and planar/unsaturated environments."),
        ("Overall conclusion", "The calculated descriptors should be interpreted together with the 2D connectivity and generated 3D geometry to understand the molecule's composition, bonding, polarity, flexibility and structural character."),
    ]


# ------------------------------------------------------------------
# 3D HELPERS
# ------------------------------------------------------------------
def show_3d(smiles, style, surface=False, height=450):
    view = py3Dmol.view(width=700, height=height)
    view.addModel(molblock_3d(smiles), "mol")
    styles = {
        "Ball and Stick": {"stick": {"radius": 0.15}, "sphere": {"scale": 0.3}},
        "Stick": {"stick": {"radius": 0.2}},
        "Space Filling": {"sphere": {}},
        "Wireframe": {"line": {}},
        "Cartoon": {"stick": {"radius": 0.08, "colorscheme": "greenCarbon"},
                    "sphere": {"scale": 0.18, "colorscheme": "greenCarbon"}},
    }
    view.setStyle({}, styles[style])
    if surface:
        view.addSurface(py3Dmol.VDW, {"opacity": 0.6, "color": "white"})
    view.setBackgroundColor("white")
    view.zoomTo()
    components.html(view._make_html(), height=height + 10)


CPK = {"H": "#FFFFFF", "C": "#333333", "N": "#3050F8", "O": "#E8261A", "S": "#E6C200",
       "P": "#FF8000", "F": "#90E050", "Cl": "#1FF01F", "Br": "#A62929"}
VDW = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "P": 1.8, "F": 1.47, "Cl": 1.75, "Br": 1.85}


def render_3d_png(smiles, style="Ball and Stick"):
    """Static picture of the 3D geometry (used inside PDF reports)."""
    mol = Chem.MolFromMolBlock(molblock_3d(smiles), removeHs=False)
    pos = mol.GetConformer().GetPositions()
    sym = [a.GetSymbol() for a in mol.GetAtoms()]
    fig = plt.figure(figsize=(5, 4), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    lw = {"Ball and Stick": 1.5, "Stick": 5, "Space Filling": 0, "Wireframe": 0.8, "Cartoon": 2.5}[style]
    if lw:
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            ax.plot(*zip(pos[i], pos[j]), color="#666666", linewidth=lw)
    if style != "Wireframe":
        for p, s in zip(pos, sym):
            if style == "Space Filling":
                size = (VDW.get(s, 1.7) * 90) ** 1.5 / 10
            elif style == "Stick":
                size = 12
            else:
                size = 45 if s != "H" else 12
            ax.scatter(*p, s=size, color=CPK.get(s, "#FF69B4"), edgecolors="black",
                       linewidths=0.5, depthshade=False)
    ax.set_axis_off()
    ax.set_title(f"3D Molecular Structure ({style.replace(' ', '-')})", fontsize=8, fontweight="bold")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ------------------------------------------------------------------
# PDF REPORT (layout follows the reference PDF)
# ------------------------------------------------------------------
NAVY = colors.HexColor("#1F3864")
BLUE = colors.HexColor("#2F5597")
LIGHT = colors.HexColor("#E8EEF7")


def rl_image(png, width):
    w, h = PILImage.open(io.BytesIO(png)).size
    return RLImage(io.BytesIO(png), width=width, height=width * h / w)


def build_report_pdf(smiles_list, style, name, reg, title):
    P = lambda n, **k: ParagraphStyle(n, fontName=k.pop("fontName", FONT), **k)
    s_title = P("t", fontName=FONT_B, fontSize=18, leading=22, alignment=TA_CENTER, textColor=NAVY, spaceAfter=10)
    s_h1 = P("h1", fontName=FONT_B, fontSize=13, leading=16, textColor=NAVY, spaceAfter=6)
    s_h2 = P("h2", fontName=FONT_B, fontSize=11.5, leading=14, textColor=NAVY, spaceBefore=10, spaceAfter=5)
    s_body = P("b", fontSize=9, leading=12)
    s_cell = P("c", fontSize=7.5, leading=9.5)
    s_cellb = P("cb", fontName=FONT_B, fontSize=7.5, leading=9.5, textColor=colors.white)
    s_bul = P("bl", fontSize=8.5, leading=11.5, leftIndent=12, bulletIndent=2, spaceAfter=3)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Cheminformatics Virtual Laboratory Report")
    story = [
        Paragraph("CHEMINFORMATICS VIRTUAL LABORATORY", s_title),
        Paragraph(title, s_h1),
        Paragraph(f"Student: {escape(name or 'Not provided')} &nbsp;&nbsp;&nbsp; "
                  f"Registration Number: {escape(reg or 'Not provided')}", s_body),
        Spacer(1, 4),
        Paragraph(f"Number of molecules analyzed: {len(smiles_list)}", s_body),
    ]

    for n, smi in enumerate(smiles_list, 1):
        mol = get_mol(smi)
        c = calc(mol)
        story.append(Paragraph(f"Molecule {n}: Molecular Information", s_h2))
        info = [["Input SMILES", smi], ["Canonical SMILES", Chem.MolToSmiles(mol)],
                ["Molecular Formula", c["formula"]], ["Molecular Weight", f"{c['mw']:.3f} g/mol"]]
        t = Table([[Paragraph(escape(a), s_body), Paragraph(escape(b), s_body)] for a, b in info],
                  colWidths=[4.5 * cm, 12.5 * cm])
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                               ("BACKGROUND", (0, 0), (0, -1), LIGHT),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story += [t, Paragraph("Molecular Structures", s_h2)]

        img2d = rl_image(pil_bytes(draw_2d(mol, (500, 350))), 8 * cm)
        img3d = rl_image(render_3d_png(smi, style), 8 * cm)
        st_ = Table([[img2d, img3d]], colWidths=[8.5 * cm, 8.5 * cm])
        st_.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        story += [st_, Paragraph("Molecular Properties and Interpretation", s_h2)]

        data = [[Paragraph("Property", s_cellb), Paragraph("Value", s_cellb), Paragraph("Interpretation", s_cellb)]]
        for p, v, i in interpretation_rows(mol):
            data.append([Paragraph(escape(p), s_cell), Paragraph(escape(v), s_cell), Paragraph(escape(i), s_cell)])
        pt = Table(data, colWidths=[3.8 * cm, 2.4 * cm, 10.8 * cm], repeatRows=1)
        pt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                                ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story += [pt, Paragraph("Key Interpretation", s_h2)]
        for head, text in key_interpretation(mol):
            story.append(Paragraph(f"<b>{head}:</b> {escape(text)}", s_bul, bulletText="•"))
        if n < len(smiles_list):
            story.append(PageBreak())

    story += [PageBreak(), Paragraph("Python Libraries Used and Their Importance", s_h1)]
    libs = [
        ("Streamlit", "Creates the interactive web app, input controls and PDF download interface."),
        ("RDKit", "Parses SMILES, validates structures, creates molecular drawings and 3D conformers, and calculates molecular descriptors."),
        ("Pandas", "Organizes molecular properties in tables and supports CSV export."),
        ("Matplotlib and NumPy", "Process molecular coordinates and generate a clear 2D projection of 3D geometry for the report."),
        ("py3Dmol", "Displays interactive 3D molecular structures in the web application."),
        ("ReportLab", "Builds the downloadable PDF with tables, figures and explanatory text."),
        ("io, os, random", "Handle in-memory PDF data, files and quiz option shuffling."),
    ]
    ldata = [[Paragraph("Library", s_cellb), Paragraph("Importance in this project", s_cellb)]]
    ldata += [[Paragraph(a, s_cell), Paragraph(b, s_cell)] for a, b in libs]
    lt = Table(ldata, colWidths=[4 * cm, 13 * cm], repeatRows=1)
    lt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                            ("BACKGROUND", (0, 0), (-1, 0), BLUE)]))
    code = (
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors\n\n"
        "smiles_list = [\"CCO\", \"c1ccccc1\"]  # Add more SMILES strings here\n\n"
        "for smiles in smiles_list:\n"
        "    mol = Chem.MolFromSmiles(smiles)\n"
        "    if mol is None:\n"
        "        print(\"Invalid SMILES:\", smiles)\n"
        "        continue\n"
        "    print(\"SMILES:\", smiles)\n"
        "    print(\"Formula:\", rdMolDescriptors.CalcMolFormula(mol))\n"
        "    print(\"Molecular weight:\", Descriptors.MolWt(mol))\n"
        "    print(\"LogP:\", Crippen.MolLogP(mol))"
    )
    story += [lt, Paragraph("Illustrative Python Code", s_h1),
              Preformatted(code, ParagraphStyle("code", fontName="Courier", fontSize=8, leading=10)),
              Spacer(1, 8),
              Paragraph("This compact code example demonstrates processing multiple SMILES in one run. "
                        "The complete application includes additional code for visualization, descriptor "
                        "interpretation, student information and PDF generation. Calculated descriptors are "
                        "computational results and should be interpreted appropriately.", s_body)]
    doc.build(story)
    return buf.getvalue()


# ------------------------------------------------------------------
# SESSION STATE + SHARED MOLECULE PICKER
# ------------------------------------------------------------------
st.session_state.setdefault("smiles", "CCO")
st.session_state.setdefault("quiz_score", None)


def load_example(text_key, ex_key):
    smi = EXAMPLES[st.session_state[ex_key]]
    st.session_state[text_key] = smi
    st.session_state["smiles"] = smi


def molecule_picker(prefix, label):
    """Text box + example dropdown. The molecule is shared across pages."""
    text_key, ex_key = f"{prefix}_smiles", f"{prefix}_example"
    st.session_state.setdefault(text_key, st.session_state["smiles"])
    st.text_input(label, key=text_key)
    st.session_state["smiles"] = st.session_state[text_key]
    st.selectbox("Choose Example Molecule", list(EXAMPLES), key=ex_key)
    st.button("Load Example", key=f"{prefix}_load", on_click=load_example, args=(text_key, ex_key))
    mol = get_mol(st.session_state["smiles"])
    if mol is None:
        st.error("❌ Invalid SMILES. Please check and try again.")
    return mol


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.title("🧪 Virtual Lab")
    page = st.radio("Navigation", PAGES)
    st.divider()
    st.subheader("👩‍🎓 Student Information")
    st.text_input("Student Name", key="student_name")
    st.text_input("Registration Number", key="student_reg")
    st.subheader("🎓 Learning Objectives")
    st.write("Students can learn:")
    st.markdown("- SMILES notation\n- Molecular representation\n- 2D visualization\n"
                "- 3D visualization\n- Molecular descriptors\n- Structure-property relationships")

student_name = st.session_state.get("student_name", "")
student_reg = st.session_state.get("student_reg", "")

# ------------------------------------------------------------------
# HOME
# ------------------------------------------------------------------
if page == PAGES[0]:
    if os.path.exists("assets/srm_logo.png"):
        st.image("assets/srm_logo.png", use_container_width=True)
    st.markdown("<div style='text-align:center;font-size:70px'>🧪</div>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center'>Cheminformatics Virtual Laboratory</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align:center;color:#1f5a99'>{INSTITUTE}</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center'>Faculty of Engineering and Technology | School of Sciences | "
                "Division of Chemistry</p>", unsafe_allow_html=True)
    st.divider()
    st.title("Welcome to the Virtual Laboratory")
    st.write("This interactive virtual laboratory helps students understand important concepts in "
             "cheminformatics and computational chemistry.")
    st.header("🧪 What can students do?")
    for item in ["Enter any SMILES notation", "Validate molecular structures", "Generate 2D structures",
                 "Generate interactive 3D structures", "Select different molecular visualization styles",
                 "Calculate complete molecular properties", "Download molecular reports",
                 "Compare multiple molecules", "Complete an assessment"]:
        st.markdown(f"✅ {item}")
    st.header("Example Molecules")
    st.dataframe(pd.DataFrame({"Molecule": list(EXAMPLES), "SMILES": list(EXAMPLES.values())}),
                 hide_index=True, use_container_width=True)
    cols = st.columns(3)
    for i, (n, smi) in enumerate(EXAMPLES.items()):
        cols[i % 3].image(draw_2d(Chem.MolFromSmiles(smi), (300, 220)), caption=n)
    st.divider()
    st.header("👩‍💻 Developed by")
    for n, r in DEVELOPERS:
        st.markdown(f"**{n}**  \nRegistration Number: {r}")
    st.write(f"Students of {INSTITUTE}")
    st.info("📧 For queries, contact:\n\n" + "\n\n".join(CONTACTS))
    st.divider()
    st.caption("🧪 Cheminformatics Virtual Laboratory | Developed using Python, Streamlit and RDKit")

# ------------------------------------------------------------------
# THEORY
# ------------------------------------------------------------------
elif page == PAGES[1]:
    st.title("📖 Introduction to Cheminformatics")
    st.header("What is Cheminformatics?")
    st.write("Cheminformatics combines:")
    st.markdown("- Chemistry ⚗️\n- Computer Science 💻\n- Data Analysis 📊")
    st.write("It uses computational techniques to store, process, visualize and analyze chemical information.")
    st.header("SMILES")
    st.write("**SMILES** means:")
    st.subheader("Simplified Molecular Input Line Entry System")
    st.write("It represents a molecular structure using a text string.")
    st.write("Examples:")
    st.table(pd.DataFrame({"Molecule": ["Ethanol", "Benzene", "Acetic Acid"],
                           "SMILES": ["CCO", "c1ccccc1", "CC(=O)O"]}))
    st.header("Molecular Descriptors")
    st.write("Molecular descriptors provide numerical information about a chemical structure.")
    st.write("Examples include:")
    st.markdown("- Molecular Weight\n- LogP\n- TPSA\n- Hydrogen Bond Donors\n"
                "- Hydrogen Bond Acceptors\n- Rotatable Bonds")
    st.header("3D Molecular Models")
    st.write("Different visualization models help students understand the molecular geometry.")
    st.subheader("🟢 Ball and Stick")
    st.write("Shows atoms as spheres and bonds as sticks.")
    st.subheader("🟡 Stick Model")
    st.write("Emphasizes chemical bonding.")
    st.subheader("🔵 Space Filling")
    st.write("Shows the approximate molecular volume.")
    st.subheader("⚫ Wireframe")
    st.write("Provides a simplified representation of the molecule.")

# ------------------------------------------------------------------
# MOLECULAR VISUALIZATION
# ------------------------------------------------------------------
elif page == PAGES[2]:
    st.title("🧬 Molecular Visualization Laboratory")
    st.info("Enter a new SMILES or load an example. The selected molecule will be used across the laboratory pages.")
    mol = molecule_picker("viz", "Enter SMILES")
    if mol:
        smiles = st.session_state["smiles"].strip()
        st.success("✅ Valid Molecular Structure!")
        st.header("2D Molecular Structure")
        st.image(draw_2d(mol), use_container_width=True)
        st.header("Molecular Information")
        c = calc(mol)
        st.markdown(f"**Input SMILES:** {smiles}")
        st.markdown(f"**Canonical SMILES:** {Chem.MolToSmiles(mol)}")
        st.markdown(f"**Formula:** {c['formula']}")
        st.markdown(f"**Heavy Atoms:** `{c['heavy']}`")
        st.markdown(f"**Total Atoms (including H):** `{c['total_atoms']}`")
        st.markdown(f"**Heavy-Atom Bonds:** `{c['hbonds']}`")
        st.markdown(f"**Sigma (σ) Bonds:** `{c['sigma']}`")
        st.markdown(f"**Pi (π) Bonds:** `{c['pi']}`")
        st.markdown(f"**Total Bonds (σ + π):** `{c['sigma'] + c['pi']}`")
        st.divider()
        st.header("🌐 Interactive 3D Molecular Structure")
        style = st.selectbox("Select Molecular Model", MODELS, key="viz_style")
        show_3d(smiles, style)

# ------------------------------------------------------------------
# MOLECULAR PROPERTIES REPORT
# ------------------------------------------------------------------
elif page == PAGES[3]:
    st.title("📊 Complete Molecular Properties Report")
    st.info("Enter any valid SMILES notation. The system will automatically generate a complete molecular properties report.")
    st.header("Select or Enter Your Molecule")
    mol = molecule_picker("rep", "Enter New SMILES")
    if mol:
        smiles = st.session_state["smiles"].strip()
        st.caption("Once you enter a valid SMILES or load an example, the complete 2D/3D structures "
                   "and molecular properties below are generated for the same molecule.")
        st.success("✅ Valid Molecular Structure Detected!")
        st.header("🧬 2D Molecular Structure")
        st.image(draw_2d(mol), use_container_width=True)

        c = calc(mol)
        st.header("📋 Basic Information")
        st.markdown(f"**Input SMILES:** {smiles}")
        st.markdown(f"**Canonical SMILES:** {Chem.MolToSmiles(mol)}")
        st.markdown(f"**Molecular Formula:** {c['formula']}")
        st.markdown(f"**Atoms:** `{c['atoms']}`")
        st.markdown(f"**Heavy Atoms:** `{c['heavy']}`")
        st.markdown(f"**Heavy-Atom Bonds:** `{c['hbonds']}`")
        st.markdown(f"**Sigma (σ) Bonds:** `{c['sigma']}`")
        st.markdown(f"**Pi (π) Bonds:** `{c['pi']}`")
        st.markdown(f"**Total Bonds (σ + π):** `{c['sigma'] + c['pi']}`")
        st.divider()

        st.header("📊 Important Molecular Properties")
        st.metric("Molecular Weight", f"{c['mw']:.3f} g/mol")
        st.metric("LogP", f"{c['logp']:.3f}")
        st.metric("TPSA", f"{c['tpsa']:.2f} Å²")
        st.metric("Ring Count", c["rings"])
        st.metric("H-Bond Donors", c["hbd"])
        st.metric("H-Bond Acceptors", c["hba"])
        st.metric("Rotatable Bonds", c["rotb"])
        st.metric("Fraction Csp3", f"{c['fsp3']:.3g}" if c["fsp3"] not in (0, 1) else f"{c['fsp3']:.1f}")
        st.divider()

        st.header("📑 Complete Molecular Property Table")
        table = full_table_df(mol, smiles)
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.download_button("⬇️ Download Properties Report (CSV)", table.to_csv(index=False),
                           file_name="molecular_properties_report.csv", mime="text/csv")

        pdf_style = st.session_state.get("rep_style", "Ball and Stick")
        if st.button("📄 Generate Molecular PDF Report"):
            with st.spinner("Generating PDF with 3D model..."):
                st.session_state["rep_pdf"] = build_report_pdf(
                    [smiles], pdf_style, student_name, student_reg, "MOLECULAR PROPERTIES REPORT")
        if "rep_pdf" in st.session_state:
            st.download_button("⬇️ Download Molecular Report PDF", st.session_state["rep_pdf"],
                               file_name="molecular_report.pdf", mime="application/pdf")
        st.divider()

        st.header("🌐 3D Molecular Visualization")
        style = st.selectbox("Select 3D Model", MODELS, key="rep_style")
        surface = st.checkbox("Show van der Waals Surface")
        if st.button("🚀 Generate 3D Molecular Model"):
            st.session_state["rep_show3d"] = True
        if st.session_state.get("rep_show3d"):
            show_3d(smiles, style, surface)

# ------------------------------------------------------------------
# STRUCTURE-PROPERTY ANALYSIS
# ------------------------------------------------------------------
elif page == PAGES[4]:
    st.title("📈 Structure–Property Analysis")
    st.write("Compare the properties of multiple molecules. Enter one molecule per line in the format:")
    st.markdown("**Molecule Name, SMILES**")
    default = "\n".join(f"{n},{s}" for n, s in list(EXAMPLES.items())[:5])
    text = st.text_area("Enter Molecules", default, height=180)
    rows = []
    for line in text.splitlines():
        if "," not in line:
            continue
        n, smi = [x.strip() for x in line.split(",", 1)]
        m = get_mol(smi)
        if m is None:
            st.warning(f"Skipped invalid SMILES: {n}")
            continue
        c = calc(m)
        viol = sum([c["mw"] > 500, c["logp"] > 5, c["hbd"] > 5, c["hba"] > 10])
        rows.append({"Molecule": n, "MW": round(c["mw"], 3), "LogP": round(c["logp"], 3),
                     "TPSA": round(c["tpsa"], 2), "HBD": c["hbd"], "HBA": c["hba"],
                     "RotB": c["rotb"], "Rings": c["rings"], "Heavy Atoms": c["heavy"],
                     "Fsp3": round(c["fsp3"], 3), "MR": round(c["mr"], 2),
                     "Lipinski Violations": viol, "SMILES": smi})
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.header("📋 Comparison Table")
        st.download_button("⬇️ Download Comparison Table (CSV)", df.to_csv(index=False),
                           file_name="comparison_table.csv", mime="text/csv")
        st.header("📊 Property Chart")
        prop = st.selectbox("Choose property", [c for c in df.columns if c not in ("Molecule", "SMILES")])
        st.bar_chart(df.set_index("Molecule")[prop])
        st.header("MW vs LogP")
        st.scatter_chart(df, x="MW", y="LogP")
        st.caption("Observation: as polarity (TPSA, HBD, HBA) increases, LogP generally decreases, "
                   "showing how structure controls lipophilicity and solubility.")

# ------------------------------------------------------------------
# ASSESSMENT
# ------------------------------------------------------------------
elif page == PAGES[5]:
    st.title("📝 Cheminformatics Assessment")
    # (question, correct answer, [wrong answers])
    QUIZ = [
        ("What is the full form of SMILES?", "Simplified Molecular Input Line Entry System",
         ["Standard Molecular Information Language Encoding System",
          "Simple Molecular Identification and Labeling System"]),
        ("What does a SMILES string represent?", "A molecular structure using text",
         ["A spectroscopy technique", "A laboratory instrument"]),
        ("Which SMILES represents ethanol?", "CCO", ["c1ccccc1", "CC(=O)O"]),
        ("Which SMILES represents benzene?", "c1ccccc1", ["CCO", "O=C=O"]),
        ("In SMILES, parentheses are mainly used to indicate:", "Branches", ["Molecular weight", "Atom colour"]),
        ("Which property is commonly associated with molecular lipophilicity?", "LogP", ["TPSA", "HBD"]),
        ("What does TPSA describe?", "Topological Polar Surface Area",
         ["Total Pi Surface Area", "Thermal Property Surface Analysis"]),
        ("What does HBA mean?", "Hydrogen Bond Acceptor", ["Heavy Bond Atom", "Hydrogen Bond Analysis"]),
        ("What does HBD mean?", "Hydrogen Bond Donor", ["Heavy Bond Descriptor", "Hydrogen Bond Distance"]),
        ("A double bond contains how many pi (π) bonds?", "1", ["2", "0"]),
        ("A triple bond contains how many pi (π) bonds?", "2", ["1", "3"]),
        ("Which model is useful for visualizing approximate molecular volume?", "Space Filling",
         ["Wireframe only", "Text SMILES"]),
        # ---- additional questions ----
        ("Which SMILES represents acetic acid?", "CC(=O)O", ["CCO", "Oc1ccccc1"]),
        ("Which SMILES represents phenol?", "Oc1ccccc1", ["c1ccccc1", "CC(=O)O"]),
        ("In SMILES, the symbol '=' represents a:", "Double bond", ["Single bond", "Triple bond"]),
        ("Lowercase atom symbols such as 'c' in SMILES usually indicate:", "Aromatic atoms",
         ["Charged atoms", "Isotopes"]),
        ("A negative LogP value suggests the molecule is:", "Hydrophilic", ["Lipophilic", "Radioactive"]),
        ("Lipinski's Rule of Five suggests molecular weight below:", "500 g/mol", ["100 g/mol", "1000 g/mol"]),
        ("Rotatable bonds mainly describe:", "Molecular flexibility", ["Molecular colour", "Boiling point"]),
        ("A fully saturated molecule such as ethanol has a Fraction Csp3 of:", "1.0", ["0.0", "0.5"]),
        ("Which Python library is used to parse SMILES and calculate descriptors?", "RDKit",
         ["Pygame", "Flask"]),
        ("Which 3D model shows atoms as spheres and bonds as sticks?", "Ball and Stick",
         ["Space Filling", "Wireframe"]),
        ("How many sigma (σ) bonds are present in every single bond?", "1", ["2", "0"]),
        ("Cheminformatics combines chemistry with:", "Computer science and data analysis",
         ["Only geology", "Only astronomy"]),
    ]
    answers, options = [], []
    for i, (q, right, wrong) in enumerate(QUIZ):
        opts = [right] + wrong
        random.Random(i).shuffle(opts)  # fixed shuffle so options stay stable between reruns
        options.append(opts)
        answers.append(st.radio(f"Q{i + 1}. {q}", opts, index=None, key=f"quiz_{i}"))
    if st.button("Submit Assessment"):
        score = sum(1 for a, (_, right, _) in zip(answers, QUIZ) if a == right)
        st.session_state["quiz_score"] = f"{score} / {len(QUIZ)}"
        (st.success if score >= len(QUIZ) * 0.6 else st.warning)(f"Your score: {score} / {len(QUIZ)}")
        wrong_list = [(i, q, r) for i, ((q, r, _), a) in enumerate(zip(QUIZ, answers)) if a != r]
        if wrong_list:
            st.subheader("Review")
            for i, q, r in wrong_list:
                st.write(f"Q{i + 1}. {q} → correct answer: **{r}**")

# ------------------------------------------------------------------
# FINAL REPORT
# ------------------------------------------------------------------
else:
    st.title("📄 Final Student Molecular Analysis Report")
    st.write("Enter one SMILES string per line. The report will include every valid molecule in a single downloadable PDF.")
    name = st.text_input("Student Name for Report", key="final_name")
    reg = st.text_input("Registration Number for Report", key="final_reg")
    text = st.text_area("Enter SMILES (one molecule per line)", "CCO", height=180,
                        help="Example: CCO, c1ccccc1, CC(=O)O")
    style = st.selectbox("3D model used in the report", MODELS, key="final_style")

    valid = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if get_mol(line) is None:
            st.warning(f"Invalid SMILES skipped: {line}")
        else:
            valid.append(line)

    if valid:
        st.success(f"{len(valid)} valid molecule(s) ready for the report.")
        summary = []
        for i, smi in enumerate(valid, 1):
            c = calc(get_mol(smi))
            summary.append({"No.": i, "Input SMILES": smi, "Molecular Formula": c["formula"],
                            "Molecular Weight": round(c["mw"], 3)})
        st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)

        if st.button("📄 Generate Single PDF for All Molecules"):
            with st.spinner("Building combined report..."):
                st.session_state["final_pdf"] = build_report_pdf(
                    valid, style, name, reg, "MULTI-MOLECULE STUDENT ANALYSIS REPORT")
            st.success("✅ Combined molecular report generated successfully!")
        if "final_pdf" in st.session_state:
            st.download_button("⬇️ Download Combined Molecular Report PDF", st.session_state["final_pdf"],
                               file_name="combined_molecular_report.pdf", mime="application/pdf")
    else:
        st.info("Enter at least one valid SMILES to continue.")

    st.divider()
    st.caption("🧪 Cheminformatics Virtual Laboratory | Developed using Python, Streamlit and RDKit")
