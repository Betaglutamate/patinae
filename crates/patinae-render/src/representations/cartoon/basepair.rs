//! Watson-Crick base-pair detection for nucleic acid cartoons.
//!
//! `CartoonType::NucleicRect` draws each strand of a double-stranded
//! nucleic acid as its own independent flat ribbon (see `tessellation.rs`
//! doc comment) -- there is no rung/ladder geometry connecting paired
//! bases across strands, unlike PyMOL's nucleic acid cartoon. This module
//! is the first half of closing that gap: a pure geometric heuristic that
//! finds candidate base pairs from atom coordinates alone, with no
//! chemistry beyond purine/pyrimidine classification.
//!
//! Detection is CPU-only and produces atom-index pairs; how those pairs
//! get turned into rendered geometry (reusing the `stick`
//! representation's capsule pipeline, most likely -- see the module docs
//! there) is left to a follow-up change, and intentionally out of scope
//! here so the detection algorithm can be reviewed and tested on its own.
//!
//! ## Method
//!
//! For every nucleic acid residue, take its `C1'` position (attachment
//! point to the base) and, if it's a purine or pyrimidine, the position
//! of the ring nitrogen that donates/accepts the central Watson-Crick
//! hydrogen bond (`N1` for purines, `N3` for pyrimidines -- present in
//! all four canonical WC pairs, so this doesn't need to distinguish A-T
//! from G-C).
//!
//! Candidates are grouped by their originating polymer subchain. Only
//! cross-subchain pairs are considered: backbone-adjacent residues within
//! one strand already sit well inside the rejection distance, and
//! restricting to cross-strand avoids spurious matches from same-strand
//! tertiary contacts (e.g. an RNA hairpin) that this MVP doesn't attempt
//! to model.
//!
//! A pair is accepted when:
//! 1. `C1'-C1'` distance falls in the canonical B-form range
//!    ([`C1_DISTANCE_RANGE`]), and
//! 2. both residues have a WC nitrogen and its distance is under
//!    [`WC_HBOND_MAX_DISTANCE`] (residues without a classified WC atom --
//!    e.g. unrecognized/generic bases -- only screen on `C1'-C1'`).
//!
//! Matching is greedy mutual-nearest-neighbor: among all candidates
//! passing the checks above, a residue only pairs with its single best
//! match, and only if that match's own best match points back to it.
//! This avoids one base claiming multiple partners in a bulge/mismatch.
//!
//! Known limitations (acceptable for this MVP; not attempted): intra-strand
//! pairing (hairpins, pseudoknots), triplexes/quadruplexes, and explicit
//! base-identity checks (A only pairs with T/U, G only with C) -- the
//! geometry alone is a strong enough filter in practice for canonical
//! duplexes, but a mismatched pair with the right geometry would still be
//! reported.

use lin_alg::f32::Vec3;
use patinae_mol::{AtomIndex, ChainView, CoordSet, ObjectMolecule};

/// Canonical B-form `C1'-C1'` distance is ~10.4 A; allow some slack for
/// real (non-idealized) structures.
const C1_DISTANCE_RANGE: (f32, f32) = (9.0, 11.5);
/// Purine N1 <-> pyrimidine N3 hydrogen-bond distance, canonical ~2.8-2.9 A;
/// allow slack for distorted/lower-resolution structures.
const WC_HBOND_MAX_DISTANCE: f32 = 3.5;

/// One detected base pair, as atom indices into the owning molecule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BasePair {
    pub c1_a: AtomIndex,
    pub c1_b: AtomIndex,
    /// WC nitrogen atoms, if both residues had one classified.
    pub wc: Option<(AtomIndex, AtomIndex)>,
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    subchain: u32,
    c1: (AtomIndex, Vec3),
    wc: Option<(AtomIndex, Vec3)>,
}

/// `true` for standard/modified purines (A, G and known modified forms).
fn is_purine(resn: &str) -> bool {
    matches!(
        patinae_mol::nucleotide_to_char(resn).map(|c| c.to_ascii_uppercase()),
        Some('A') | Some('G')
    )
}

/// `true` for standard/modified pyrimidines (C, T, U and known modified forms).
fn is_pyrimidine(resn: &str) -> bool {
    matches!(
        patinae_mol::nucleotide_to_char(resn).map(|c| c.to_ascii_uppercase()),
        Some('C') | Some('T') | Some('U')
    )
}

fn gather_candidates(molecule: &ObjectMolecule, coord_set: &CoordSet) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    let mut subchain_idx: u32 = 0;

    let chains: Vec<ChainView<'_>> = molecule.chains().collect();
    for chain in &chains {
        for subchain in chain.polymer_subchains() {
            let mut any_nucleic = false;
            let mut local = Vec::new();
            for residue in subchain.residues() {
                if !residue.is_nucleic() {
                    continue;
                }
                any_nucleic = true;
                let Some((c1_idx, _)) = residue.find_by_name("C1'") else {
                    continue;
                };
                let Some(c1_pos) = coord_set.get_atom_coord(c1_idx) else {
                    continue;
                };
                let wc_name = if is_purine(residue.resn()) {
                    Some("N1")
                } else if is_pyrimidine(residue.resn()) {
                    Some("N3")
                } else {
                    None
                };
                let wc = wc_name.and_then(|name| {
                    let (idx, _) = residue.find_by_name(name)?;
                    let pos = coord_set.get_atom_coord(idx)?;
                    Some((idx, pos))
                });
                local.push(Candidate {
                    subchain: subchain_idx,
                    c1: (c1_idx, c1_pos),
                    wc,
                });
            }
            if any_nucleic {
                subchain_idx += 1;
            }
            candidates.extend(local);
        }
    }
    candidates
}

/// Distance between two candidates used for the confirm/screen checks:
/// WC nitrogen distance when both have one, else `C1'-C1'` distance.
fn confirm_distance(a: &Candidate, b: &Candidate) -> Option<f32> {
    match (a.wc, b.wc) {
        (Some((_, pa)), Some((_, pb))) => Some((pa - pb).magnitude()),
        _ => None,
    }
}

fn is_candidate_pair(a: &Candidate, b: &Candidate) -> bool {
    if a.subchain == b.subchain {
        return false;
    }
    let c1_dist = (a.c1.1 - b.c1.1).magnitude();
    if c1_dist < C1_DISTANCE_RANGE.0 || c1_dist > C1_DISTANCE_RANGE.1 {
        return false;
    }
    match confirm_distance(a, b) {
        Some(d) => d <= WC_HBOND_MAX_DISTANCE,
        // Neither/only-one side has a classified WC atom -- accept on the
        // C1'-C1' screen alone.
        None => true,
    }
}

/// Score used to pick the *best* match for a candidate: smaller is better.
/// Prefers the WC nitrogen distance (more specific) over `C1'-C1'`.
fn match_score(a: &Candidate, b: &Candidate) -> f32 {
    confirm_distance(a, b).unwrap_or_else(|| (a.c1.1 - b.c1.1).magnitude())
}

fn best_match(idx: usize, candidates: &[Candidate]) -> Option<usize> {
    let a = &candidates[idx];
    candidates
        .iter()
        .enumerate()
        .filter(|(j, b)| *j != idx && is_candidate_pair(a, b))
        .min_by(|(_, b1), (_, b2)| {
            match_score(a, b1)
                .partial_cmp(&match_score(a, b2))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(j, _)| j)
}

/// Detect Watson-Crick base pairs across all nucleic acid subchains in
/// `molecule`. See the module docs for the method and its limitations.
pub fn detect_base_pairs(molecule: &ObjectMolecule, coord_set: &CoordSet) -> Vec<BasePair> {
    let candidates = gather_candidates(molecule, coord_set);
    let best: Vec<Option<usize>> = (0..candidates.len())
        .map(|i| best_match(i, &candidates))
        .collect();

    let mut pairs = Vec::new();
    for i in 0..candidates.len() {
        let Some(j) = best[i] else { continue };
        if j <= i {
            // Either not mutual, or already emitted from the other side
            // (mutual pairs are only emitted once, when i < j).
            continue;
        }
        if best[j] != Some(i) {
            continue;
        }
        let a = &candidates[i];
        let b = &candidates[j];
        pairs.push(BasePair {
            c1_a: a.c1.0,
            c1_b: b.c1.0,
            wc: match (a.wc, b.wc) {
                (Some((wa, _)), Some((wb, _))) => Some((wa, wb)),
                _ => None,
            },
        });
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use patinae_mol::{AtomBuilder, AtomFlags, AtomResidue, MoleculeBuilder};
    use std::sync::Arc;

    /// One residue's worth of test atoms: C1' plus its WC nitrogen.
    struct ResidueFixture {
        chain: &'static str,
        resn: &'static str,
        resv: i32,
        c1: [f32; 3],
        wc_name: &'static str,
        wc: [f32; 3],
    }

    fn build_duplex(residues: &[ResidueFixture]) -> ObjectMolecule {
        let mut b = MoleculeBuilder::new("duplex");
        for r in residues {
            let res = Arc::new(AtomResidue::from_parts(r.chain, r.resn, r.resv, ' ', ""));

            let mut c1 = AtomBuilder::new().name("C1'").element_symbol("C").build();
            c1.residue = res.clone();
            c1.state.flags |= AtomFlags::NUCLEIC | AtomFlags::POLYMER;
            b = b.add_atom(c1, Vec3::new(r.c1[0], r.c1[1], r.c1[2]));

            let mut wc = AtomBuilder::new()
                .name(r.wc_name)
                .element_symbol("N")
                .build();
            wc.residue = res;
            wc.state.flags |= AtomFlags::NUCLEIC | AtomFlags::POLYMER;
            b = b.add_atom(wc, Vec3::new(r.wc[0], r.wc[1], r.wc[2]));
        }
        b.build()
    }

    /// Idealized canonical B-form ATGC / GCAT duplex, generated via PyMOL's
    /// `cmd.fnab('ATGC', mode='DNA', form='B', dbl_helix=1)` -- the same
    /// method used elsewhere in this project to build idealized duplexes.
    /// Real coordinates, not hand-waved, so the expected pairing below is
    /// grounded in actual canonical B-DNA geometry.
    fn idealized_atgc_duplex() -> ObjectMolecule {
        build_duplex(&[
            ResidueFixture {
                chain: "A",
                resn: "DA",
                resv: 1,
                c1: [1.256415, 5.500146, -0.007610],
                wc_name: "N1",
                wc: [-0.498585, 0.670146, -0.007610],
            },
            ResidueFixture {
                chain: "A",
                resn: "DT",
                resv: 2,
                c1: [4.032189, 3.355099, -3.382609],
                wc_name: "N3",
                wc: [0.535407, 2.181277, -3.521610],
            },
            ResidueFixture {
                chain: "A",
                resn: "DG",
                resv: 3,
                c1: [5.017010, -0.011838, -6.757609],
                wc_name: "N1",
                wc: [-0.077671, 0.099257, -6.748610],
            },
            ResidueFixture {
                chain: "A",
                resn: "DC",
                resv: 4,
                c1: [3.834711, -3.314610, -10.132609],
                wc_name: "N3",
                wc: [1.571203, -0.372133, -10.263609],
            },
            ResidueFixture {
                chain: "B",
                resn: "DG",
                resv: -4,
                c1: [-6.330181, -0.011837, -10.164610],
                wc_name: "N1",
                wc: [-1.235499, 0.099257, -10.173609],
            },
            ResidueFixture {
                chain: "B",
                resn: "DC",
                resv: -3,
                c1: [-5.147882, -3.314611, -6.789610],
                wc_name: "N3",
                wc: [-2.884373, -0.372134, -6.658610],
            },
            ResidueFixture {
                chain: "B",
                resn: "DA",
                resv: -2,
                c1: [-2.250060, -5.291673, -3.414610],
                wc_name: "N1",
                wc: [-0.830881, -0.352559, -3.414610],
            },
            ResidueFixture {
                chain: "B",
                resn: "DT",
                resv: -1,
                c1: [1.256415, -5.187853, -0.039610],
                wc_name: "N3",
                wc: [-0.882585, -2.182853, 0.099390],
            },
        ])
    }

    #[test]
    fn detects_all_four_pairs_of_idealized_duplex() {
        let mol = idealized_atgc_duplex();
        let coord = mol.current_coord_set().expect("coord set").clone();
        let pairs = detect_base_pairs(&mol, &coord);
        assert_eq!(pairs.len(), 4, "expected one pair per base, got {pairs:?}");

        // Antiparallel Watson-Crick pairing: A's 5'->3' (resv 1..4) pairs
        // with B's 3'->5' (resv -1..-4) in the same order (A1-B(-1),
        // A2-B(-2), A3-B(-3), A4-B(-4)).
        let mut resv_pairs: Vec<(i32, i32)> = pairs
            .iter()
            .map(|p| {
                let ra = mol.get_atom(p.c1_a).unwrap().residue.resv;
                let rb = mol.get_atom(p.c1_b).unwrap().residue.resv;
                if ra > 0 {
                    (ra, rb)
                } else {
                    (rb, ra)
                }
            })
            .collect();
        resv_pairs.sort();
        assert_eq!(resv_pairs, vec![(1, -1), (2, -2), (3, -3), (4, -4)]);
    }

    #[test]
    fn no_pairs_for_single_strand() {
        // Same coordinates, chain A only -- no partner strand exists, so
        // there should be nothing to pair (and nothing must panic).
        let mol = build_duplex(&[
            ResidueFixture {
                chain: "A",
                resn: "DA",
                resv: 1,
                c1: [1.256415, 5.500146, -0.007610],
                wc_name: "N1",
                wc: [-0.498585, 0.670146, -0.007610],
            },
            ResidueFixture {
                chain: "A",
                resn: "DT",
                resv: 2,
                c1: [4.032189, 3.355099, -3.382609],
                wc_name: "N3",
                wc: [0.535407, 2.181277, -3.521610],
            },
        ]);
        let coord = mol.current_coord_set().expect("coord set").clone();
        assert!(detect_base_pairs(&mol, &coord).is_empty());
    }
}
