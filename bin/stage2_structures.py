import requests, numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser, MMCIFParser

GENES = {
    "MUC2":   "P98088",
    "ORM2":   "P19652",
    "CST1":   "P01037",
    "LRP1B":  "Q9NZR2",
    "CLEC3A": "Q8IX30",
    "CASP14": "P31944",
}

def download_structure(gene, uid):
    """Try multiple AF2 URL formats until one works."""
    urls = [
        # Current AF2 EBI endpoint (v4)
        f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v4.pdb",
        # Fallback v3
        f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v3.pdb",
        # RCSB PDB direct (some proteins are deposited here)
        f"https://files.rcsb.org/download/{uid}.pdb",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and "ATOM" in r.text:
                fname = f"{gene}_af2.pdb"
                with open(fname, "w") as f:
                    f.write(r.text)
                print(f"  {gene}: downloaded from {url.split('/')[4]}")
                return fname
        except Exception:
            continue
    return None

def get_plddt(pdb_path):
    """Extract pLDDT (B-factor column) from AF2 PDB."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("prot", pdb_path)
    plddt = []
    for model in struct.get_models():
        for chain in model.get_chains():
            for res in chain.get_residues():
                if res.has_id("CA"):
                    plddt.append(res["CA"].get_bfactor())
            if plddt:
                return np.array(plddt)
    return np.array(plddt)

# Download and parse all structures
print("Downloading AlphaFold2 structures...")
plddt_data = {}
for gene, uid in GENES.items():
    fname = download_structure(gene, uid)
    if fname and os.path.exists(fname):
        plddt = get_plddt(fname)
        if len(plddt) > 0:
            plddt_data[gene] = plddt
            conf = (plddt > 70).mean() * 100
            print(f"  {gene:8s}: {len(plddt):5d} residues | "
                  f"mean pLDDT={plddt.mean():.1f} | conf={conf:.0f}%")
        else:
            print(f"  {gene}: downloaded but no CA atoms found — check file")
    else:
        print(f"  {gene}: download failed — using simulated data for demo")
        # Simulate realistic pLDDT for demo purposes
        np.random.seed(list(GENES.keys()).index(gene))
        n = {"MUC2":5179,"ORM2":201,"CST1":141,"LRP1B":4599,
             "CLEC3A":228,"CASP14":242}[gene]
        base = {"MUC2":55,"ORM2":85,"CST1":88,"LRP1B":72,
                "CLEC3A":82,"CASP14":86}[gene]
        sim = np.clip(np.random.normal(base, 15, n), 0, 100)
        plddt_data[gene] = sim
        conf = (sim > 70).mean() * 100
        print(f"  {gene:8s}: {n:5d} residues (simulated) | "
              f"mean={sim.mean():.1f} | conf={conf:.0f}%")

# Check if we got any real data
real_genes = []
for gene in GENES:
    fname = f"{gene}_af2.pdb"
    if os.path.exists(fname):
        size = os.path.getsize(fname)
        atom_lines = sum(1 for l in open(fname) if l.startswith("ATOM"))
        print(f"  {gene}_af2.pdb: {size} bytes, {atom_lines} ATOM lines")
        real_genes.append(gene)

# Plot
colors = {"MUC2":"#7F77DD","ORM2":"#1D9E75","CST1":"#D85A30",
          "LRP1B":"#E24B4A","CLEC3A":"#378ADD","CASP14":"#BA7517"}

fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, gene in zip(axes.flat, GENES):
    plddt = plddt_data[gene]
    # Subsample long proteins for plot clarity
    if len(plddt) > 1000:
        idx = np.linspace(0, len(plddt)-1, 1000).astype(int)
        plddt_plot = plddt[idx]
    else:
        plddt_plot = plddt

    ax.fill_between(range(len(plddt_plot)), plddt_plot,
                    alpha=0.2, color=colors[gene])
    ax.plot(plddt_plot, color=colors[gene], lw=1.5)
    ax.axhline(70, color="#712B13", ls="--", lw=1.2,
               label="confident (>70)")
    ax.axhline(50, color="gray", ls=":", lw=1,
               label="low conf (<50)")
    conf = (plddt > 70).mean() * 100
    note = " [real]" if gene in real_genes else " [sim]"
    ax.set_title(f"{gene}{note}\n"
                 f"mean={plddt.mean():.0f}  conf={conf:.0f}%",
                 fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Residue position")
    ax.set_ylabel("pLDDT")
    ax.legend(fontsize=7, loc="lower right")

plt.suptitle(
    "AlphaFold2 structural confidence — 6 breast cancer survival biomarkers\n"
    "Identified by SHAP analysis of TCGA BRCA RNA-seq (1,233 patients)",
    fontsize=12, y=1.01
)
plt.tight_layout()
plt.savefig("biomarker_plddt.png", dpi=130, bbox_inches="tight")
print("\nSaved: biomarker_plddt.png")

# Summary table
print(f"\n{'Gene':<10} {'Residues':>8} {'Mean pLDDT':>11} "
      f"{'Conf%':>7}  {'Source':<6}  Interpretation")
print("-" * 75)
for gene, plddt in plddt_data.items():
    conf = (plddt > 70).mean() * 100
    src  = "real" if gene in real_genes else "sim "
    if   plddt.mean() > 80: interp = "Excellent design target"
    elif plddt.mean() > 70: interp = "Good — mostly structured"
    elif plddt.mean() > 50: interp = "Mixed — partial disorder"
    else:                    interp = "Disordered — interaction hub"
    print(f"{gene:<10} {len(plddt):>8} {plddt.mean():>11.1f} "
          f"{conf:>6.0f}%  {src}    {interp}")
