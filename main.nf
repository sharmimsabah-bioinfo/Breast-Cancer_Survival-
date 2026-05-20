#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// ─────────────────────────────────────────────────────────────────────────────
// BRCA Survival → Protein Design Pipeline
// Sharmim Sultana — Flagship PI Co-Op Project
//
// Stages:
//   1. Survival prediction (Deep MLP + RL ensemble)
//   2. AlphaFold2 structural analysis (6 genes IN PARALLEL)
//   3. Flow matching sequence design (per gene)
//   4. MD stability simulation (top 5 designs per gene)
//   5. Final report
//
// Run locally : nextflow run main.nf -profile local
// Run on AWS  : nextflow run main.nf -profile aws
// Resume crash: nextflow run main.nf -profile local -resume
// ─────────────────────────────────────────────────────────────────────────────

params.genes_csv    = "data/genes.csv"
params.n_designs    = 100
params.plddt_thresh = 70
params.outdir       = "results"

// ── Process 1: Survival prediction ───────────────────────────────────────────
process SURVIVAL_PREDICTION {
    publishDir "${params.outdir}/survival", mode: 'copy'
    memory '8 GB'
    cpus 4

    output:
    path "biomarkers.csv",        emit: biomarkers
    path "*.png",                 emit: figures
    path "model_summary.txt",     emit: summary

    script:
    """
    python bin/stage1_survival.py --outdir .
    """
}

// ── Process 2: AlphaFold2 structure analysis (runs 6x IN PARALLEL) ───────────
process STRUCTURE_ANALYSIS {
    publishDir "${params.outdir}/structures", mode: 'copy'
    tag "$gene_name"
    memory '4 GB'
    cpus 2

    input:
    tuple val(gene_name), val(uniprot_id)

    output:
    tuple val(gene_name), val(uniprot_id),
          path("${gene_name}_af2.pdb"),
          path("${gene_name}_plddt.csv"), emit: structures

    script:
    """
    python bin/stage2_structures.py \\
        --gene ${gene_name} \\
        --uniprot ${uniprot_id} \\
        --outdir .
    """
}

// ── Process 3: Flow matching design (conditional on pLDDT) ───────────────────
process FLOW_MATCHING_DESIGN {
    publishDir "${params.outdir}/designs/${gene_name}", mode: 'copy'
    tag "$gene_name"
    label 'gpu'
    memory '16 GB'
    cpus 8

    input:
    tuple val(gene_name), val(uniprot_id),
          path(pdb_file), path(plddt_csv)

    output:
    tuple val(gene_name),
          path("${gene_name}_designs.csv"),
          path("${gene_name}_rl_history.csv"),
          path("${gene_name}_results.png"), emit: designs

    script:
    """
    python bin/stage3_full_pipeline.py \\
        --gene ${gene_name} \\
        --pdb ${pdb_file} \\
        --n_designs ${params.n_designs} \\
        --plddt_thresh ${params.plddt_thresh} \\
        --outdir .
    """
}

// ── Process 4: MD stability simulation ───────────────────────────────────────
process MD_STABILITY {
    publishDir "${params.outdir}/md_results", mode: 'copy'
    tag "$gene_name"
    memory '8 GB'
    cpus 4

    input:
    tuple val(gene_name), path(designs_csv),
          path(rl_history), path(design_fig)

    output:
    tuple val(gene_name), path("${gene_name}_md_results.csv"), emit: md

    script:
    """
    python bin/md_stability.py \\
        --gene ${gene_name} \\
        --designs ${designs_csv} \\
        --top_n 5 \\
        --duration_ns 10 \\
        --outdir .
    """
}

// ── Process 5: Final summary report ──────────────────────────────────────────
process GENERATE_REPORT {
    publishDir "${params.outdir}/report", mode: 'copy'

    input:
    path biomarkers
    path "designs/*"
    path "md_results/*"

    output:
    path "pipeline_summary.png"
    path "pipeline_summary.html"

    script:
    """
    python bin/generate_report.py \\
        --biomarkers ${biomarkers} \\
        --designs_dir designs/ \\
        --md_dir md_results/ \\
        --outdir .
    """
}

// ── Workflow: wires all processes together ────────────────────────────────────
workflow {

    // Read gene list: emits [gene_name, uniprot_id] tuples
    genes_ch = Channel
        .fromPath(params.genes_csv)
        .splitCsv(header: true)
        .map { row -> tuple(row.gene, row.uniprot_id) }

    // Stage 1: survival (runs once, produces biomarker gene list)
    survival_out = SURVIVAL_PREDICTION()

    // Stage 2: structure analysis for all 6 genes IN PARALLEL
    struct_out = STRUCTURE_ANALYSIS(genes_ch)

    // Stage 3: flow matching design
    // Filter: only design genes with mean pLDDT > threshold
    design_input = struct_out.structures.filter { gene, uid, pdb, plddt_csv ->
        def lines = plddt_csv.readLines()
        lines.size() > 1 && lines[1].split(',')[1].toFloat() > params.plddt_thresh
    }
    design_out = FLOW_MATCHING_DESIGN(design_input)

    // Stage 4: MD stability on top designs
    md_out = MD_STABILITY(design_out.designs)

    // Stage 5: final report (collects all results)
    GENERATE_REPORT(
        survival_out.biomarkers,
        design_out.designs.map { it[1] }.collect(),
        md_out.md.map { it[1] }.collect()
    )
}
