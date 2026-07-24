//! Nucleic acid base-pair "rung" rendering.
//!
//! Turns [`super::basepair::detect_base_pairs`]'s output into actual
//! geometry: one capsule per detected base pair, connecting the two
//! strands' `C1'` atoms (the canonical "ladder" look of a double-helix
//! cartoon).
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
//! catalog entry in `representations/catalog.rs`), so rungs
//! automatically show/hide alongside the cartoon representation with no
//! separate "show" plumbing needed.

use patinae_mol::{CoordSet, DirtyFlags, ObjectMolecule, RepMask};
use patinae_settings::ResolvedSettings;

use crate::picking::RepKind;
use crate::pipelines::stick::{StickParams, StickParamsLayout};
use crate::render_input::RenderObjectInput;
use crate::representations::cartoon::basepair::detect_base_pairs;
use crate::representations::stick::StickInstance;
use crate::representations::{BuildCtx, Representation};

/// Capsule radius for rungs -- thinner than the default stick radius
/// (0.25), since a rung is an accent, not a bond.
const RUNG_RADIUS: f32 = 0.2;

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

/// Resolve detected base pairs to their `C1'` world positions, as plain
/// tuples `(atom_a, atom_b, pos_a, pos_b)` -- device-free so it's directly
/// unit-testable (see [`tests::resolves_one_pair_with_matching_positions`]).
///
/// A pair is dropped unless *both* `C1'` atoms currently have the cartoon
/// representation visible (`repr.visible_reps` -- the same per-atom bit
/// `cartoon.wgsl`'s fragment shader checks via `scene_atom_in_rep` to
/// discard hidden cartoon pixels). Rungs are drawn through the stick
/// pipeline directly, bypassing both that shader and the stick
/// compute-build's own bond/atom visibility filtering, so without this
/// check they'd ignore "hide cartoon" entirely.
fn resolve_pairs(
    molecule: &ObjectMolecule,
    coord_set: &CoordSet,
) -> Vec<(u32, u32, [f32; 3], [f32; 3])> {
    detect_base_pairs(molecule, coord_set)
        .iter()
        .filter_map(|p| {
            let atom_a = molecule.get_atom(p.c1_a)?;
            let atom_b = molecule.get_atom(p.c1_b)?;
            if !atom_a.repr.visible_reps.is_visible(RepMask::CARTOON)
                || !atom_b.repr.visible_reps.is_visible(RepMask::CARTOON)
            {
                return None;
            }
            let pa = coord_set.get_atom_coord(p.c1_a)?;
            let pb = coord_set.get_atom_coord(p.c1_b)?;
            Some((
                p.c1_a.as_u32(),
                p.c1_b.as_u32(),
                [pa.x, pa.y, pa.z],
                [pb.x, pb.y, pb.z],
            ))
        })
        .collect()
}

/// Build one [`StickInstance`] rung per resolved pair. Device-free, unit-testable.
fn rung_instances_from_pairs(resolved: &[(u32, u32, [f32; 3], [f32; 3])]) -> Vec<StickInstance> {
    resolved
        .iter()
        .map(|(a, b, pa, pb)| StickInstance {
            p0_radius: [pa[0], pa[1], pa[2], RUNG_RADIUS],
            p1_pad: [pb[0], pb[1], pb[2], 0.0],
            groups: [*a, *b],
            _pad: [0, 0],
        })
        .collect()
}

/// Cheap signature over the base pairs found, so an unrelated rebuild
/// (e.g. color-only) doesn't re-run detection/upload every frame.
fn hash_pairs(pairs: &[(u32, u32, [f32; 3], [f32; 3])]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    pairs.len().hash(&mut hasher);
    for (a, b, pa, pb) in pairs {
        a.hash(&mut hasher);
        b.hash(&mut hasher);
        for c in pa.iter().chain(pb.iter()) {
            c.to_bits().hash(&mut hasher);
        }
    }
    hasher.finish()
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

        // Unlike CartoonRep, rung visibility is baked into `resolved` at
        // build time (see `resolve_pairs`'s per-atom CARTOON check) rather
        // than gated later in a fragment shader -- so, unintuitively,
        // `DirtyFlags::VISIBILITY` (part of CartoonRep's `LUT_ONLY_MASK`
        // fast path) *must* trigger a real rebuild here, or `hide cartoon`
        // would leave stale rungs on screen. No fast path: just recompute
        // and rely on `hash_pairs`/`last_signature` below to skip the
        // upload when nothing relevant actually changed.
        let resolved = resolve_pairs(input.molecule, input.coord_set);
        let sig = hash_pairs(&resolved);
        if self.last_signature == Some(sig) {
            return;
        }
        self.last_signature = Some(sig);

        if resolved.is_empty() {
            self.instance_buf = None;
            self.count = 0;
            return;
        }

        let instances = rung_instances_from_pairs(&resolved);

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
    use patinae_mol::{AtomBuilder, AtomFlags, AtomResidue, MoleculeBuilder};
    use std::sync::Arc;

    #[allow(clippy::too_many_arguments)]
    fn add_residue(
        mut b: MoleculeBuilder,
        chain: &str,
        resn: &str,
        resv: i32,
        c1: [f32; 3],
        wc_name: &str,
        wc: [f32; 3],
        cartoon_visible: bool,
    ) -> MoleculeBuilder {
        let res = Arc::new(AtomResidue::from_parts(chain, resn, resv, ' ', ""));
        let mut c1_atom = AtomBuilder::new().name("C1'").element_symbol("C").build();
        c1_atom.residue = res.clone();
        c1_atom.state.flags |= AtomFlags::NUCLEIC | AtomFlags::POLYMER;
        if cartoon_visible {
            c1_atom.repr.visible_reps.set_visible(RepMask::CARTOON);
        }
        b = b.add_atom(c1_atom, lin_alg::f32::Vec3::new(c1[0], c1[1], c1[2]));

        let mut wc_atom = AtomBuilder::new().name(wc_name).element_symbol("N").build();
        wc_atom.residue = res;
        wc_atom.state.flags |= AtomFlags::NUCLEIC | AtomFlags::POLYMER;
        if cartoon_visible {
            wc_atom.repr.visible_reps.set_visible(RepMask::CARTOON);
        }
        b.add_atom(wc_atom, lin_alg::f32::Vec3::new(wc[0], wc[1], wc[2]))
    }

    /// A1/B(-1) from the idealized ATGC/GCAT duplex used in
    /// `basepair::tests` -- a known, hand-verified Watson-Crick pair (real
    /// coordinates from PyMOL's `fnab`, not hand-waved).
    fn one_pair_molecule() -> ObjectMolecule {
        one_pair_molecule_with_visibility(true, true)
    }

    fn one_pair_molecule_with_visibility(a_visible: bool, b_visible: bool) -> ObjectMolecule {
        let b = MoleculeBuilder::new("pair");
        let b = add_residue(
            b,
            "A",
            "DA",
            1,
            [1.256415, 5.500146, -0.007610],
            "N1",
            [-0.498585, 0.670146, -0.007610],
            a_visible,
        );
        let b = add_residue(
            b,
            "B",
            "DT",
            -1,
            [1.256415, -5.187853, -0.039610],
            "N3",
            [-0.882585, -2.182853, 0.099390],
            b_visible,
        );
        b.build()
    }

    #[test]
    fn resolves_one_pair_with_matching_positions() {
        let mol = one_pair_molecule();
        let coord = mol.current_coord_set().expect("coord set").clone();
        let resolved = resolve_pairs(&mol, &coord);
        assert_eq!(resolved.len(), 1);
        let (a, b, pa, pb) = resolved[0];
        // C1' of residue A1 is atom index 0, C1' of residue B(-1) is atom
        // index 2 (C1', WC atom pairs added per-residue).
        assert_eq!((a, b), (0, 2));
        assert!((pa[1] - 5.500146).abs() < 1e-4);
        assert!((pb[1] - (-5.187853)).abs() < 1e-4);
    }

    #[test]
    fn builds_one_rung_instance_with_expected_radius_and_groups() {
        let mol = one_pair_molecule();
        let coord = mol.current_coord_set().expect("coord set").clone();
        let resolved = resolve_pairs(&mol, &coord);
        let instances = rung_instances_from_pairs(&resolved);
        assert_eq!(instances.len(), 1);
        let inst = instances[0];
        assert_eq!(inst.p0_radius[3], RUNG_RADIUS);
        assert_eq!(inst.groups, [0, 2]);
        assert!((inst.p0_radius[1] - 5.500146).abs() < 1e-4);
        assert!((inst.p1_pad[1] - (-5.187853)).abs() < 1e-4);
    }

    #[test]
    fn no_pairs_yields_no_instances() {
        let resolved: Vec<(u32, u32, [f32; 3], [f32; 3])> = Vec::new();
        assert!(rung_instances_from_pairs(&resolved).is_empty());
    }

    #[test]
    fn pair_dropped_when_cartoon_hidden_for_either_side() {
        // Regression test: rungs are drawn through the stick pipeline
        // directly, bypassing both cartoon.wgsl's per-atom `discard` and
        // the stick compute-build's own visibility filtering, so without
        // resolve_pairs' explicit CARTOON check, `hide cartoon` would
        // leave stale rungs on screen.
        let both_hidden = one_pair_molecule_with_visibility(false, false);
        let coord = both_hidden.current_coord_set().expect("coord set").clone();
        assert!(resolve_pairs(&both_hidden, &coord).is_empty());

        let one_hidden = one_pair_molecule_with_visibility(true, false);
        let coord = one_hidden.current_coord_set().expect("coord set").clone();
        assert!(resolve_pairs(&one_hidden, &coord).is_empty());

        let both_visible = one_pair_molecule_with_visibility(true, true);
        let coord = both_visible.current_coord_set().expect("coord set").clone();
        assert_eq!(resolve_pairs(&both_visible, &coord).len(), 1);
    }
}
