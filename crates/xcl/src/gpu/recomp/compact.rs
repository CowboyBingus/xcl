//! Compact form of a very large program: every distinct block text is emitted once.
//!
//! DXC inlines every function of a cs_6_x target, so HLSL helpers do not share code: the
//! 3,281 buffer loads of a retail program become 3,281 copies of the format conversion. Here
//! the program is cut into blocks (at the leaders, and around every instruction whose HLSL is
//! long), each instruction is emitted by `emit::Em` exactly as in `translate`, with its
//! temporaries numbered from t1, and blocks with the same text share one `case` of a
//! dispatch loop. A branch goes to the block at an offset from its own (part of the text), so
//! the table (`XCL_BT`) only gives every block, in address order, its case; falling through
//! goes to the next block. The block index is the same in every lane (branch conditions are
//! wave-uniform), as `pc` of the guarded block chain is, so the split-wave64 halves walk the
//! same blocks.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;

use super::{emit, prepare, rdna2, Mode, Options, Prepared, Translation, LOOP_GUARD};

/// Programs with fewer instructions keep the normal form (`compact_wanted`).
pub const COMPACT_MIN_INSTRUCTIONS: usize = 20_000;

/// An instruction whose HLSL is longer than this (bytes) is a block of its own, so the blocks
/// around it can still be shared when their other instructions differ.
const HEAVY: usize = 600;

/// Largest table (32-bit words). A static array is constant data for the driver: on an
/// RTX 5070 Ti, CreateComputePipelineState took 12,288 words and failed with E_OUTOFMEMORY
/// at 15,360.
const MAX_TABLE_WORDS: usize = 8192;

/// A class assigned on its first occurrence equals the number of classes seen so far. Store
/// only later references; the bitset and its per-word rank make every block a random lookup.
struct RepeatTable {
    repeats: Vec<u32>,
    prefix: Vec<u32>,
    refs: Vec<u32>,
}

fn repeat_table(class: &[u32], bits: u32) -> RepeatTable {
    let mut repeats = vec![0u32; class.len().div_ceil(32)];
    let mut prefix = Vec::with_capacity(repeats.len());
    let mut refs = Vec::new();
    let mut next = 0u32;
    for (i, &c) in class.iter().enumerate() {
        if i % 32 == 0 {
            prefix.push(refs.len() as u32);
        }
        if c == next {
            next += 1;
        } else {
            debug_assert!(c < next);
            repeats[i / 32] |= 1 << (i % 32);
            refs.push(c);
        }
    }
    let mut packed = vec![0u32; (refs.len() * bits as usize).div_ceil(32)];
    for (i, c) in refs.into_iter().enumerate() {
        let bit = i * bits as usize;
        let (word, shift) = (bit / 32, bit % 32);
        packed[word] |= c << shift;
        if shift + bits as usize > 32 {
            packed[word + 1] |= c >> (32 - shift);
        }
    }
    RepeatTable { repeats, prefix, refs: packed }
}

fn write_array(body: &mut String, name: &str, words: &[u32]) {
    let _ = writeln!(body, "    static const uint {name}[{}] = {{", words.len());
    for row in words.chunks(16) {
        let row: Vec<String> = row.iter().map(|v| format!("{v:#x}u")).collect();
        let _ = writeln!(body, "        {},", row.join(", "));
    }
    body.push_str("    };\n");
}

/// The compact form pays: at least `COMPACT_MIN_INSTRUCTIONS` instructions, and the distinct
/// blocks (equal code, branch targets aside) hold under 30 % of them. A function of the code
/// alone; nothing is emitted.
pub fn compact_wanted(code: &[u8]) -> bool {
    // an instruction has at least 4 bytes
    if code.len() < 4 * COMPACT_MIN_INSTRUCTIONS {
        return false;
    }
    let mut insts = Vec::new();
    let mut off = 0usize;
    while off + 4 <= code.len() {
        let Ok(d) = rdna2::decode(&code[off..], off as u64) else { return false };
        off += d.len;
        insts.push(d);
    }
    if insts.len() < COMPACT_MIN_INSTRUCTIONS {
        return false;
    }
    rdna2::resolve_far_jumps(&mut insts);
    let mut leaders = BTreeSet::new();
    for d in &insts {
        let t = rdna2::branch_target(d);
        if t.is_some() || d.inst.flags & rdna2::F_TERMINATOR != 0 {
            leaders.insert(d.pc + d.len as u64);
        }
        leaders.extend(t);
    }
    // a block's key: its code, without the last instruction's bytes when that one branches
    let mut seen: HashSet<(&[u8], &str)> = HashSet::new();
    let (mut distinct, mut first, mut start) = (0usize, 0usize, 0usize);
    for (k, d) in insts.iter().enumerate() {
        let end = d.pc as usize + d.len;
        if k + 1 < insts.len() && !leaders.contains(&(end as u64)) {
            continue;
        }
        let key = if rdna2::branch_target(d).is_some() { (&code[start..d.pc as usize], d.inst.name) } else { (&code[start..end], "") };
        if seen.insert(key) {
            distinct += k + 1 - first;
        }
        (first, start) = (k + 1, end);
    }
    distinct * 10 < insts.len() * 3
}

/// `recomp::translate` in the compact form (same contract and output; `Translation::blocks`
/// is the number of distinct block texts). A program without branches gets `translate`'s
/// straight-line form; geometry programs (ES → GS through s_setpc) are not supported.
pub fn translate_compact(code: &[u8], opt: &Options) -> Result<Translation, String> {
    compact(code, opt, HEAVY)
}

/// One block in address order: its text, and the branch that ends it (condition, empty when
/// it always jumps; target).
struct Block {
    pc: u64,
    text: String,
    branch: Option<(String, u64)>,
}

fn block(pc: u64, mut text: String, flow: emit::Flow) -> Result<Block, String> {
    let branch = match flow {
        emit::Flow::Next => None,
        emit::Flow::Jump(t) => Some((String::new(), t)),
        emit::Flow::Cond(c, t) => Some((c, t)),
        emit::Flow::End => {
            text.push_str("            xcl_nb = 0xFFFFFFFFu;\n");
            None
        }
        emit::Flow::JumpExpr(_) => return Err(format!("+{pc:#x}: compact form: computed jump")),
    };
    Ok(Block { pc, text, branch })
}

/// The HLSL of one instruction without its comment lines; in its own scope when it declares
/// temporaries (every instruction numbers them from t1).
fn inst_text(em: &mut emit::Em, d: &rdna2::Decoded, next: u64) -> (String, emit::Flow) {
    em.reset_temps();
    let flow = em.inst(d, next);
    let temps = em.reset_temps();
    let mut s = String::new();
    for l in em.take().lines().filter(|l| !l.trim_start().starts_with("//")) {
        s.push_str(l);
        s.push('\n');
    }
    if temps > 0 && !s.is_empty() {
        s = format!("            {{\n{s}            }}\n");
    }
    (s, flow)
}

/// The blocks of the program in address order (`heavy`: see `HEAVY`).
fn blocks(insts: &[rdna2::Decoded], leaders: &BTreeSet<u64>, em: &mut emit::Em, heavy: usize) -> Result<Vec<Block>, String> {
    em.indent = 3;
    let mut out = Vec::new();
    // the open block: its address and text; `dead`: after the block's last instruction
    let mut open: Option<(u64, String)> = None;
    let mut dead = false;
    for (i, d) in insts.iter().enumerate() {
        if leaders.contains(&d.pc) {
            if let Some((pc, text)) = open.take() {
                out.push(block(pc, text, emit::Flow::Next)?);
            }
            dead = false;
        }
        if dead {
            continue;
        }
        let next = insts.get(i + 1).map(|n| n.pc).unwrap_or(u64::MAX);
        let (text, flow) = inst_text(em, d, next);
        if text.len() > heavy && matches!(flow, emit::Flow::Next) {
            if let Some((pc, text)) = open.take() {
                out.push(block(pc, text, emit::Flow::Next)?);
            }
            out.push(block(d.pc, text, flow)?);
            continue;
        }
        let (pc, mut body) = open.take().unwrap_or((d.pc, String::new()));
        body.push_str(&text);
        if matches!(flow, emit::Flow::Next) {
            open = Some((pc, body));
        } else {
            out.push(block(pc, body, flow)?);
            dead = true;
        }
    }
    if let Some((pc, text)) = open.take() {
        out.push(block(pc, text, emit::Flow::Next)?);
    }
    Ok(out)
}

fn compact(code: &[u8], opt: &Options, heavy: usize) -> Result<Translation, String> {
    if matches!(opt.mode, Mode::Geometry(_)) {
        return Err("compact form: geometry programs are not supported".into());
    }
    let Prepared { insts, leaders, branches, mut em, .. } = prepare(code, opt)?;
    if branches == 0 {
        return super::translate(code, opt);
    }
    let list = blocks(&insts, &leaders, &mut em, heavy)?;
    let n = list.len();
    let at: HashMap<u64, usize> = list.iter().enumerate().map(|(k, b)| (b.pc, k)).collect();
    // the distinct texts (with the address of the first block of each); each block's text
    let mut texts: Vec<(String, u64)> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut class = Vec::with_capacity(n);
    for (k, b) in list.into_iter().enumerate() {
        let mut text = b.text;
        if let Some((cond, t)) = b.branch {
            let to = *at.get(&t).ok_or_else(|| format!("+{:#x}: branch to {t:#x}, which starts no block", b.pc))?;
            let nb = if to >= k { format!("xcl_b + {}u", to - k) } else { format!("xcl_b - {}u", k - to) };
            let _ = if cond.is_empty() { writeln!(text, "            xcl_nb = {nb};") } else { writeln!(text, "            if ({cond}) xcl_nb = {nb};") };
        }
        let c = match index.get(&text) {
            Some(&c) => c,
            None => {
                index.insert(text.clone(), texts.len());
                texts.push((text, b.pc));
                texts.len() - 1
            }
        };
        class.push(c as u32);
    }
    // the table: the blocks' cases, `per` to a word
    let bits = (usize::BITS - texts.len().saturating_sub(1).leading_zeros()).max(1);
    let per = (32 / bits) as usize;
    let words: Vec<u32> = class.chunks(per).map(|c| c.iter().enumerate().fold(0u32, |w, (j, &v)| w | v << (j as u32 * bits))).collect();
    let mut body = String::new();
    let repeated = (words.len() > MAX_TABLE_WORDS).then(|| repeat_table(&class, bits));
    if let Some(t) = &repeated {
        let size = t.repeats.len() + t.prefix.len() + t.refs.len();
        if size > MAX_TABLE_WORDS {
            return Err(format!("compact form: {n} blocks of {} texts need a {size}-word repeated-block table (at most {MAX_TABLE_WORDS})", texts.len()));
        }
        write_array(&mut body, "XCL_RB", &t.repeats);
        write_array(&mut body, "XCL_RP", &t.prefix);
        write_array(&mut body, "XCL_RV", &t.refs);
    } else {
        write_array(&mut body, "XCL_BT", &words);
    }
    body.push_str("    uint xcl_b = 0u;\n    [loop] for (;;) {\n        uint xcl_nb = xcl_b + 1u;\n");
    let mask = (1u64 << bits) - 1;
    if repeated.is_some() {
        body.push_str("        uint xcl_group = xcl_b / 32u;\n        uint xcl_pos = xcl_b % 32u;\n");
        body.push_str("        uint xcl_flags = XCL_RB[xcl_group];\n");
        body.push_str("        uint xcl_before = XCL_RP[xcl_group] + countbits(xcl_flags & ((1u << xcl_pos) - 1u));\n");
        body.push_str("        uint xcl_case = xcl_b - xcl_before;\n");
        body.push_str("        if ((xcl_flags & (1u << xcl_pos)) != 0u) {\n");
        let _ = writeln!(body, "            uint xcl_bit = xcl_before * {bits}u;");
        body.push_str("            uint xcl_word = xcl_bit / 32u;\n            uint xcl_shift = xcl_bit % 32u;\n");
        body.push_str("            xcl_case = XCL_RV[xcl_word] >> xcl_shift;\n");
        let _ = writeln!(body, "            if (xcl_shift + {bits}u > 32u) xcl_case |= XCL_RV[xcl_word + 1u] << (32u - xcl_shift);");
        let _ = writeln!(body, "            xcl_case &= {mask:#x}u;");
        body.push_str("        }\n        switch (xcl_case / 256u) {\n");
    } else {
        let _ = writeln!(body, "        switch ((XCL_BT[xcl_b / {per}u] >> (xcl_b % {per}u * {bits}u)) & {mask:#x}u) {{");
    }
    if repeated.is_some() {
        // Keep DXC's control-flow work bounded by 256 cases per switch. The 11,909 distinct
        // cases of ActivateRays in one switch still had not compiled after ten minutes.
        for (section, chunk) in texts.chunks(256).enumerate() {
            let _ = writeln!(body, "        case {section}u: {{ switch (xcl_case % 256u) {{");
            for (k, (text, pc)) in chunk.iter().enumerate() {
                let _ = write!(body, "            case {k}u: {{ // {pc:x}\n{text}            }} break;\n");
            }
            body.push_str("            default: xcl_nb = 0xFFFFFFFFu; break;\n        } } break;\n");
        }
    } else {
        for (k, (text, pc)) in texts.iter().enumerate() {
            let _ = write!(body, "        case {k}u: {{ // {pc:x}\n{text}        }} break;\n");
        }
    }
    body.push_str("        default: xcl_nb = 0xFFFFFFFFu; break;\n        }\n");
    let _ = writeln!(body, "        if (xcl_nb >= {n}u) break;");
    // a back edge spends the loop budget (LOOP_GUARD, 65,536 per lane); once it is spent the
    // program ends, as in the guarded block chain. The structured form instead leaves the
    // loop and runs what follows.
    let _ = writeln!(body, "        if (xcl_nb <= xcl_b) {{ {LOOP_GUARD} }}");
    body.push_str("        xcl_b = xcl_nb;\n    }\n");
    let hlsl = em.finish(&body);
    let static_samplers = em.static_used.clone();
    Ok(Translation {
        hlsl,
        unsupported: em.unsupported,
        unsupported_why: em.unsupported_why,
        ignored: em.ignored,
        instructions: insts.len(),
        blocks: texts.len(),
        static_samplers,
    })
}

#[cfg(test)]
mod tests {
    use super::super::runner::{self, KernelState};
    use super::*;
    use crate::gpu::rdna2::encode;

    const V: u64 = 256; // VALU source code of v0

    type Inst<'a> = (&'a str, &'a str, &'a [(&'a str, u64)]);

    fn prog(list: &[Inst]) -> Vec<u8> {
        let mut v = Vec::new();
        for (n, e, f) in list {
            v.extend(encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}")));
        }
        v
    }

    /// A scalar loop, then three exec-guarded regions (the first two with the same code, the
    /// third skipped: no lane passes its compare).
    fn guarded() -> Vec<u8> {
        let mul: Inst = ("V_MUL_LO_U32", "ENC_VOP3", &[("VDST", 2), ("SRC0", V), ("SRC1", V)]);
        let add: Inst = ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 2), ("VSRC1", 1)]);
        let save: Inst = ("S_MOV_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 126)]);
        let restore: Inst = ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 2)]);
        prog(&[
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 128)]),             // 0: s0 = 0
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 138)]),             // 4: s1 = 10
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 1)]),  // 8: s0 += s1
            ("S_SUB_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 129)]), // 12: s1 -= 1
            ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 1), ("SSRC1", 128)]),         // 16
            ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFC)]),                 // 20: → 8
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 0)]),                // 24: v1 = s0
            save,                                                                  // 28
            ("V_CMPX_GT_U32", "ENC_VOPC", &[("SRC0", 148), ("VSRC1", 0)]),         // 32: 20 > v0
            ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 3)]),                     // 36: → 52
            mul,                                                                   // 40
            add,                                                                   // 48
            restore,                                                               // 52
            ("V_CMPX_LT_U32", "ENC_VOPC", &[("SRC0", 158), ("VSRC1", 0)]),         // 56: 30 < v0
            ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 3)]),                     // 60: → 76
            mul,                                                                   // 64
            add,                                                                   // 72
            restore,                                                               // 76
            ("V_CMPX_GT_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 0)]),         // 80: 0 > v0
            ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)]),                     // 84: → 92
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 135)]),              // 88: v4 = 7
            restore,                                                               // 92
            ("S_ENDPGM", "ENC_SOPP", &[]),                                         // 96
        ])
    }

    #[test]
    fn equal_blocks_share_one_case() {
        let code = guarded();
        let t = translate_compact(&code, &Options::default()).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        assert!(t.hlsl.contains("switch ((XCL_BT[xcl_b / ") && t.hlsl.contains("static const uint XCL_BT[1]"), "{}", t.hlsl);
        // blocks: 0, 8, 24, 40, 52, 64, 76, 88, 92; 40 and 64 are the same code
        assert_eq!(t.blocks, 8, "{}", t.hlsl);
        assert_eq!(t.hlsl.matches("v3 = select(").count(), 1, "{}", t.hlsl);
        // branches by offset: the loop's back edge, the skips over two blocks and over one
        assert!(t.hlsl.contains("xcl_nb = xcl_b + 0u;") && t.hlsl.contains("xcl_nb = xcl_b + 2u;") && t.hlsl.contains("xcl_nb = xcl_b + 1u;"), "{}", t.hlsl);
        // every instruction a block of its own: the same instruction texts are shared too
        let each = compact(&code, &Options::default(), 0).unwrap();
        assert!(each.blocks < each.instructions && each.blocks > t.blocks, "{}", each.hlsl);
        // no branches: the straight-line form of `translate`
        let line = prog(&[("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 128)])]);
        assert_eq!(translate_compact(&line, &Options::default()).unwrap().hlsl, super::super::translate(&line, &Options::default()).unwrap().hlsl);
        if super::super::dxc::location().is_ok() {
            for o in [Options::default(), Options { split_wave64: true, mem_windows: 4, ..Options::default() }] {
                let t = translate_compact(&code, &o).unwrap();
                super::super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
            }
        }
    }

    #[test]
    fn compact_is_wanted_for_large_repetitive_programs() {
        assert!(!compact_wanted(&guarded()));
        // 20,000 instructions: one guarded block 5,000 times, or 10,000 different blocks
        let unit = prog(&[
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 2), ("VSRC1", 1)]),
            ("V_CMPX_GT_U32", "ENC_VOPC", &[("SRC0", 148), ("VSRC1", 0)]),
            ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)]),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 135)]),
        ]);
        let same: Vec<u8> = unit.iter().copied().cycle().take(unit.len() * 5000).collect();
        assert!(compact_wanted(&same));
        assert!(!compact_wanted(&same[..same.len() - 4]));
        let mut different = Vec::new();
        for k in 0..10_000u32 {
            different.extend(encode("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 255)], Some(k)).unwrap());
            different.extend(encode("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0)], None).unwrap());
        }
        assert!(!compact_wanted(&different));
    }

    #[test]
    fn repeated_block_table_recovers_every_case() {
        // The shape of ActivateRays: 19,949 blocks, 11,909 distinct cases. Include both
        // new and repeated cases in each 32-block group and references crossing word edges.
        let mut class = Vec::with_capacity(19_949);
        let mut next = 0u32;
        for i in 0..19_949 {
            if next < 11_909 && (next == 0 || i % 5 != 0) {
                class.push(next);
                next += 1;
            } else {
                class.push((i * 137 % next as usize) as u32);
            }
        }
        assert_eq!(next, 11_909);
        let bits = 14u32;
        let t = repeat_table(&class, bits);
        assert!(t.repeats.len() + t.prefix.len() + t.refs.len() <= MAX_TABLE_WORDS);
        for (i, &expected) in class.iter().enumerate() {
            let group = i / 32;
            let pos = i % 32;
            let flags = t.repeats[group];
            let before = t.prefix[group] + (flags & ((1u32 << pos) - 1)).count_ones();
            let actual = if flags & (1 << pos) == 0 {
                i as u32 - before
            } else {
                let bit = before as usize * bits as usize;
                let (word, shift) = (bit / 32, bit % 32);
                let mut c = t.refs[word] >> shift;
                if shift + bits as usize > 32 {
                    c |= t.refs[word + 1] << (32 - shift);
                }
                c & ((1 << bits) - 1)
            };
            assert_eq!(actual, expected, "block {i}");
        }
    }

    /// The compact form ends in the same state as `translate` (both block cuts).
    #[test]
    fn gpu_compact_matches_translate() {
        match (runner::adapter(), runner::wave64_programs_supported(), super::super::dxc::location()) {
            (Ok(_), Ok(true), Ok(_)) => {}
            (a, w, d) => {
                runner::skip_test(if a.is_err() || w.is_err() { "no GPU" } else if d.is_err() { "no DXC" } else { "no wave64" });
                return;
            }
        }
        let code = guarded();
        let mem: Vec<u32> = code.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut st = KernelState::default();
        for (lane, v) in st.vgpr.iter_mut().enumerate() {
            v[0] = lane as u32;
        }
        let opt = runner::options(Mode::Kernel);
        let run = |t: Translation| {
            assert!(t.complete(), "{:?}", t.unsupported_why);
            let dxil = super::super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
            runner::run(&dxil, &st, &mem).unwrap()
        };
        let want = run(super::super::translate(&code, &opt).unwrap());
        assert_eq!((want.sgpr[0], want.vgpr[5][3], want.vgpr[40][3], want.vgpr[25][3]), (55, 80, 1655, 0));
        for heavy in [HEAVY, 0] {
            let got = run(compact(&code, &opt, heavy).unwrap());
            assert_eq!((got.sgpr, got.vgpr, got.scc), (want.sgpr.clone(), want.vgpr.clone(), want.scc), "heavy {heavy}");
        }
    }
}
