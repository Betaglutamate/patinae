//! Nucleic acid base rendering: the real sugar-phosphate backbone is
//! already drawn by `CartoonType::NucleicRect` (see `tessellation.rs`);
//! this fills in the base rings using the molecule's own atoms and real
//! covalent bonds, the same way `sticks` renders anything else. There is
//! no synthetic "this residue pairs with that one" step -- a real
//! Watson-Crick duplex places paired bases right next to each other in
//! space, so drawing every base's actual ring geometry at its actual
//! position already produces the classic ladder look. This matches how
//! PyMOL does it: the "rungs" are real atoms, not an invented connector.
//!
//! This is its own [`RepKind::NucleicLadder`], not extra geometry bolted
//! onto [`super::CartoonRep`]'s own draw calls -- reusing the cartoon
//! pipeline's slot would mean manually swapping to the stick pipeline
//! mid-`record_translucent` and back, which would corrupt the outer
//! dispatch loop's per-`RepKind` pipeline-binding cache (it only calls
//! `set_pipeline` when the kind changes between consecutive draws, see
//! `render_state/visible_flow.rs`). A separate `RepKind` lets the
//! existing dispatch loop bind the (reused, not duplicated) stick
//! pipeline correctly on its own.
//!
//! Shares [`RepMask::CARTOON`] with `RepKind::Cartoon`/`Ribbon` (see the
//! catalog entry in `representations/catalog.rs`), so bases automatically
//! show/hide alongside the cartoon representation with no separate "show"
//! plumbing needed.

use patinae_mol::{AtomFlags, CoordSet, DirtyFlags, ObjectMolecule, RepMask};
use patinae_settings::ResolvedSettings;

use crate::picking::RepKind;
use crate::pipelines::stick::{StickParams, StickParamsLayout};
use crate::render_input::RenderObjectInput;
use crate::representations::stick::StickInstance;
use crate::representations::{BuildCtx, Representation};

/// Capsule radius for base sticks -- thinner than the default stick
/// radius (0.25), to read as an accent on the cartoon rather than a
/// full atomic-detail stick rendering.
const BASE_STICK_RADIUS: f32 = 0.15;

/// Standard sugar-phosphate backbone atom names (DNA and RNA). Anything
/// else on a `NUCLEIC`-flagged atom is part of the base. Deliberately a
/// small, chemistry-agnostic exclusion list rather than an enumeration
/// of base-ring atom names, so modified/methylated bases (5mC, 5hmC,
/// 5fC, ...) work the same as standard ones with no extra cases needed.
fn is_nucleic_backbone_atom_name(name: &str) -> bool {
    matches!(
        name,
        "P" | "OP1"
            | "OP2"
            | "OP3"
            | "O1P"
            | "O2P"
            | "O3P"
            | "O5'"
            | "C5'"
            | "C4'"
            | "O4'"
            | "C3'"
            | "O3'"
            | "C2'"
            | "O2'"
            | "C1'"
    )
}

/// Resolve real bonds worth drawing as base sticks: both atoms nucleic,
/// not a pure backbone-backbone bond (the ribbon already represents
/// that), and both currently cartoon-visible (see the module docs on
/// `RepMask::CARTOON` sharing -- the stick pipeline reused here has no
/// per-atom visibility gate of its own, so it must be applied here).
/// Device-free so it's directly unit-testable.
fn resolve_base_bonds(
    molecule: &ObjectMolecule,
    coord_set: &CoordSet,
) -> Vec<(u32, u32, [f32; 3], [f32; 3])> {
    molecule
        .bonds()
        .filter_map(|bond| {
            let a = molecule.get_atom(bond.atom1)?;
            let b = molecule.get_atom(bond.atom2)?;
            if !a.state.flags.contains(AtomFlags::NUCLEIC)
                || !b.state.flags.contains(AtomFlags::NUCLEIC)
            {
                return None;
            }
            if is_nucleic_backbone_atom_name(&a.name) && is_nucleic_backbone_atom_name(&b.name) {
                return None;
            }
            if !a.repr.visible_reps.is_visible(RepMask::CARTOON)
                || !b.repr.visible_reps.is_visible(RepMask::CARTOON)
            {
                return None;
            }
            let pa = coord_set.get_atom_coord(bond.atom1)?;
            let pb = coord_set.get_atom_coord(bond.atom2)?;
            Some((
                bond.atom1.as_u32(),
                bond.atom2.as_u32(),
                [pa.x, pa.y, pa.z],
                [pb.x, pb.y, pb.z],
            ))
        })
        .collect()
}

/// Build one [`StickInstance`] per resolved bond. Device-free, unit-testable.
fn stick_instances_from_bonds(resolved: &[(u32, u32, [f32; 3], [f32; 3])]) -> Vec<StickInstance> {
    resolved
        .iter()
        .map(|(a, b, pa, pb)| StickInstance {
            p0_radius: [pa[0], pa[1], pa[2], BASE_STICK_RADIUS],
            p1_pad: [pb[0], pb[1], pb[2], 0.0],
            groups: [*a, *b],
            _pad: [0, 0],
        })
        .collect()
}

/// Cheap signature over the resolved bonds, so an unrelated rebuild
/// (e.g. color-only) doesn't re-run resolution/upload every frame.
fn hash_bonds(bonds: &[(u32, u32, [f32; 3], [f32; 3])]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bonds.len().hash(&mut hasher);
    for (a, b, pa, pb) in bonds {
        a.hash(&mut hasher);
        b.hash(&mut hasher);
        for c in pa.iter().chain(pb.iter()) {
            c.to_bits().hash(&mut hasher);
        }
    }
    hasher.finish()
}

pub struct NucleicLadderRep {
    instance_buf: Option<wgpu::Buffer>,
    count: u32,
    render_params_buffer: wgpu::Buffer,
    render_params_bind_group: Option<wgpu::BindGroup>,
    last_signature: Option<u64>,
}

impl NucleicLadderRep {
    pub fn new(device: &wgpu::Device) -> Self {
        let render_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("patinae.nucleic_ladder.render_params"),
            size: StickParams::SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            instance_buf: None,
            count: 0,
            render_params_buffer,
            render_params_bind_group: None,
            last_signature: None,
        }
    }
}

impl Representation for NucleicLadderRep {
    fn kind(&self) -> RepKind {
        RepKind::NucleicLadder
    }

    fn build(
        &mut self,
        input: &RenderObjectInput,
        settings: &ResolvedSettings,
        _dirty: DirtyFlags,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let s = input.object_settings.as_ref().unwrap_or(settings);
        if !s.cartoon.nucleic_ladder {
            self.instance_buf = None;
            self.count = 0;
            self.last_signature = None;
            return;
        }

        // No fast path on `_dirty`: bond visibility is baked into
        // `resolved` here (unlike CartoonRep, whose visibility gating
        // lives entirely in a fragment shader and needs no CPU rebuild),
        // so skipping rebuild on visibility-only dirty would mean `hide
        // cartoon` never re-resolves. hash_bonds/last_signature below
        // still skips the redundant upload when nothing changed.
        let resolved = resolve_base_bonds(input.molecule, input.coord_set);
        let sig = hash_bonds(&resolved);
        if self.last_signature == Some(sig) {
            return;
        }
        self.last_signature = Some(sig);

        if resolved.is_empty() {
            self.instance_buf = None;
            self.count = 0;
            return;
        }

        let instances = stick_instances_from_bonds(&resolved);

        self.instance_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("patinae.nucleic_ladder.instances"),
            size: (instances.len() * std::mem::size_of::<StickInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        queue.write_buffer(
            self.instance_buf.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&instances),
        );
        self.count = instances.len() as u32;
    }

    fn is_opaque(&self) -> bool {
        true
    }

    fn casts_shadow(&self) -> bool {
        false
    }

    fn record_compute_build(&mut self, ctx: &mut BuildCtx<'_>) -> bool {
        if self.render_params_bind_group.is_none() {
            let layout: &StickParamsLayout = &ctx.pipelines.stick_params_layout;
            let params = StickParams::default();
            ctx.queue
                .write_buffer(&self.render_params_buffer, 0, bytemuck::bytes_of(&params));
            self.render_params_bind_group =
                Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("patinae.nucleic_ladder.render_bg"),
                    layout: &layout.bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.render_params_buffer.as_entire_binding(),
                    }],
                }));
        }
        false
    }

    fn record_translucent<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.count == 0 {
            return;
        }
        let (Some(buf), Some(bg)) = (
            self.instance_buf.as_ref(),
            self.render_params_bind_group.as_ref(),
        ) else {
            return;
        };
        pass.set_bind_group(3, bg, &[]);
        pass.set_vertex_buffer(0, buf.slice(..));
        pass.draw(0..6, 0..self.count);
    }

    fn record_picking<'a>(&'a self, _pass: &mut wgpu::RenderPass<'a>) {
        // Not pickable -- catalog::picking_pipeline returns None for
        // this kind, so the picking pass never calls this anyway.
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use patinae_mol::{AtomBuilder, AtomResidue, BondOrder, MoleculeBuilder};
    use std::sync::Arc;

    /// A minimal single nucleotide with real bonds: P-O5'-C5'-C4'-...-C1'
    /// backbone (all backbone-backbone, should be excluded) and a
    /// C1'-N1 glycosidic bond plus one N1-C2 ring bond (both should be
    /// included, since N1/C2 aren't backbone atom names).
    fn one_nucleotide_molecule(cartoon_visible: bool) -> ObjectMolecule {
        let mut b = MoleculeBuilder::new("nt");
        let res = Arc::new(AtomResidue::from_parts("A", "DA", 1, ' ', ""));

        let add = |b: MoleculeBuilder, name: &str, x: f32| -> MoleculeBuilder {
            let mut atom = AtomBuilder::new().name(name).element_symbol("C").build();
            atom.residue = res.clone();
            atom.state.flags |= AtomFlags::NUCLEIC | AtomFlags::POLYMER;
            if cartoon_visible {
                atom.repr.visible_reps.set_visible(RepMask::CARTOON);
            }
            b.add_atom(atom, lin_alg::f32::Vec3::new(x, 0.0, 0.0))
        };

        b = add(b, "C1'", 0.0);
        b = add(b, "N1", 1.0);
        b = add(b, "C2", 2.0);
        let mut built = b.build();

        let idx_c1 = built.atoms_indexed().next().unwrap().0;
        let idx_n1 = built.atoms_indexed().nth(1).unwrap().0;
        let idx_c2 = built.atoms_indexed().nth(2).unwrap().0;
        let _ = built.add_bond(idx_c1, idx_n1, BondOrder::Single);
        let _ = built.add_bond(idx_n1, idx_c2, BondOrder::Aromatic);
        built
    }

    #[test]
    fn includes_glycosidic_and_ring_bonds_not_pure_backbone() {
        let mol = one_nucleotide_molecule(true);
        let coord = mol.current_coord_set().expect("coord set").clone();
        let resolved = resolve_base_bonds(&mol, &coord);
        // C1'-N1 (backbone-to-base) and N1-C2 (base-to-base) both
        // included; there is no pure backbone-backbone bond in this
        // fixture to exclude, but the base atoms are what matter here.
        assert_eq!(resolved.len(), 2);
    }

    #[test]
    fn excludes_pure_backbone_backbone_bond() {
        let mut mol = one_nucleotide_molecule(true);
        // Add a second backbone atom and a pure backbone-backbone bond.
        let mut o5 = AtomBuilder::new().name("O5'").element_symbol("O").build();
        o5.residue = Arc::new(AtomResidue::from_parts("A", "DA", 1, ' ', ""));
        o5.state.flags |= AtomFlags::NUCLEIC | AtomFlags::POLYMER;
        o5.repr.visible_reps.set_visible(RepMask::CARTOON);
        let idx_o5 = mol.add_atom(o5);
        mol.set_coord(idx_o5, 0, lin_alg::f32::Vec3::new(-1.0, 0.0, 0.0));
        let idx_c1 = mol.atoms_indexed().next().unwrap().0;
        let _ = mol.add_bond(idx_o5, idx_c1, BondOrder::Single);

        let coord = mol.current_coord_set().expect("coord set").clone();
        let resolved = resolve_base_bonds(&mol, &coord);
        // Still just the glycosidic + ring bond; O5'-C1' (both backbone
        // names) must not appear.
        assert_eq!(resolved.len(), 2);
        for (a, b, _, _) in &resolved {
            assert!(*a != idx_o5.as_u32() && *b != idx_o5.as_u32());
        }
    }

    #[test]
    fn excludes_bonds_when_cartoon_hidden() {
        let mol = one_nucleotide_molecule(false);
        let coord = mol.current_coord_set().expect("coord set").clone();
        assert!(resolve_base_bonds(&mol, &coord).is_empty());
    }

    #[test]
    fn builds_stick_instance_per_resolved_bond() {
        let mol = one_nucleotide_molecule(true);
        let coord = mol.current_coord_set().expect("coord set").clone();
        let resolved = resolve_base_bonds(&mol, &coord);
        let instances = stick_instances_from_bonds(&resolved);
        assert_eq!(instances.len(), resolved.len());
        for inst in &instances {
            assert_eq!(inst.p0_radius[3], BASE_STICK_RADIUS);
        }
    }

    #[test]
    fn no_bonds_yields_no_instances() {
        let resolved: Vec<(u32, u32, [f32; 3], [f32; 3])> = Vec::new();
        assert!(stick_instances_from_bonds(&resolved).is_empty());
    }
}
