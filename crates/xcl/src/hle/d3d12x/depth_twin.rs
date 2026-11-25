//! lab: a depth buffer the console also lets compute shaders write (ALLOW_DEPTH_STENCIL |
//! ALLOW_UNORDERED_ACCESS; the PC refuses the pair, `device_x::pc_resource_desc` keeps the
//! depth buffer) gets a twin texture with ALLOW_UNORDERED_ACCESS for the plane a UAV names:
//! the title's UAVs of the depth buffer are made of the twin, and after every compute dispatch
//! whose descriptor tables name such a view the twin is copied into the depth buffer's plane
//! (`list_x::depth_twin_flush`; the title moves these buffers through no state of its own).
//!
//! A retail title writes its material masks into the STENCIL of a 2560x1440 D16S8 buffer with a
//! compute pass (a X24_TYPELESS_G8_UINT UAV) and its lighting passes test / read that stencil;
//! it also blits cached local-light shadow maps into D16 buffers with compute (R16_UNORM UAVs).
//! With null views all of that was dropped. `XCL_DEPTH_TWIN=0`: off; `=s`: stencil twins only.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

#[derive(Clone, Copy)]
pub struct Info {
    pub twin: usize,
    /// stencil twins: the buffer the plane goes through (0: none, the formats are the same)
    pub staging: usize,
    /// subresources of the plane (mips x array), and the plane's first subresource in the depth buffer
    pub subs: u32,
    pub first: u32,
    pub fp_twin: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    pub fp_plane: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
}

struct Twin {
    _res: ID3D12Resource,
    _staging: Option<ID3D12Resource>,
    shape: (u64, u32, u16, u16, i32),
    info: Info,
}
unsafe impl Send for Twin {}

/// (depth buffer, plane) → its twin
static TWINS: Mutex<Option<HashMap<(usize, u32), Twin>>> = Mutex::new(None);
static ANY: AtomicBool = AtomicBool::new(false);
static BYTES: AtomicU64 = AtomicU64::new(0);
/// image descriptor word 0 (the slot's PC heap index) → (depth buffer, plane) its view is of
static VIEWS: Mutex<Option<HashMap<u32, (usize, u32)>>> = Mutex::new(None);
/// (depth buffer, subresource; u32::MAX: all) → the PC state the title's last barrier left
static STATES: Mutex<Option<HashMap<(usize, u32), u32>>> = Mutex::new(None);

fn mode() -> u8 {
    static M: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *M.get_or_init(|| match std::env::var("XCL_DEPTH_TWIN").as_deref().map(str::trim) {
        Ok("0") => 0,
        Ok("s") => 1,
        _ => 2,
    })
}

fn shape(d: &D3D12_RESOURCE_DESC) -> (u64, u32, u16, u16, i32) {
    (d.Width, d.Height, d.DepthOrArraySize, d.MipLevels, d.Format.0)
}

pub fn any() -> bool {
    ANY.load(Ordering::Relaxed)
}

/// The twin of the plane of the depth buffer `res` that a UAV in `view_format` (the PC's) names,
/// made now if there is none, and that plane: None when there cannot be one.
pub unsafe fn get_or_make(res: *mut c_void, view_format: i32) -> Option<(*mut c_void, u32)> {
    if mode() == 0 {
        return None;
    }
    let r = ID3D12Resource::from_raw_borrowed(&res)?;
    let d = r.GetDesc();
    if d.SampleDesc.Count != 1 || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D {
        return None;
    }
    // R32_TYPELESS / R16_TYPELESS (D32, D16): the one plane, a twin in the same format.
    // R24G8_TYPELESS / R32G8X24_TYPELESS: the stencil (views X24_TYPELESS_G8_UINT /
    // X32_TYPELESS_G8X24_UINT), an R8_UINT twin; a UAV of their depth stays a null view.
    let (plane, twin_format) = match (d.Format.0, view_format) {
        (39 | 53, _) if mode() == 2 => (0u32, d.Format),
        (44, 47) | (19, 22) if d.DepthOrArraySize == 1 && d.MipLevels == 1 => (1, DXGI_FORMAT_R8_UINT),
        _ => return None,
    };
    let mut g = TWINS.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if let Some(t) = m.get(&(res as usize, plane)) {
        if t.shape == shape(&d) {
            return Some((t.info.twin as *mut c_void, plane));
        }
    }
    let mut dev: Option<ID3D12Device> = None;
    r.GetDevice(&mut dev).ok()?;
    let dev = dev?;
    let td = D3D12_RESOURCE_DESC { Alignment: 0, Format: twin_format, Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, ..d };
    let mut twin: Option<ID3D12Resource> = None;
    let made = dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, None, &mut twin);
    let Some(twin) = twin.filter(|_| made.is_ok()) else {
        crate::xlog_warn!("D3D12.X: depth twin: no texture for {}x{}x{} format {}: {made:?}", d.Width, d.Height, d.DepthOrArraySize, d.Format.0);
        return None;
    };
    let subs = d.DepthOrArraySize as u32 * d.MipLevels as u32;
    let first = plane * subs;
    let (mut fp_twin, mut fp_plane) = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(), D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default());
    let (mut total_twin, mut total_plane) = (0u64, 0u64);
    dev.GetCopyableFootprints(&td, 0, 1, 0, Some(&mut fp_twin), None, None, Some(&mut total_twin));
    dev.GetCopyableFootprints(&d, first, 1, 0, Some(&mut fp_plane), None, None, Some(&mut total_plane));
    let mut staging: Option<ID3D12Resource> = None;
    if plane == 1 {
        if fp_twin.Footprint.RowPitch != fp_plane.Footprint.RowPitch || total_plane == 0 {
            crate::xlog_warn!("D3D12.X: depth twin: stencil footprints differ ({:?} / {:?})", fp_twin.Footprint, fp_plane.Footprint);
            return None;
        }
        let bd = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: total_twin.max(total_plane),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let made = dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut staging);
        if made.is_err() || staging.is_none() {
            crate::xlog_warn!("D3D12.X: depth twin: no staging buffer: {made:?}");
            return None;
        }
    }
    let texel = match twin_format.0 {
        39 => 4,
        53 => 2,
        _ => 1,
    };
    let bytes = d.Width * d.Height as u64 * subs as u64 * texel;
    let total = BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    let info = Info { twin: twin.as_raw() as usize, staging: staging.as_ref().map_or(0, |s| s.as_raw() as usize), subs, first, fp_twin, fp_plane };
    m.insert((res as usize, plane), Twin { _res: twin, _staging: staging, shape: shape(&d), info });
    ANY.store(true, Ordering::Relaxed);
    if m.len() <= 8 || m.len().is_power_of_two() || plane == 1 {
        crate::xlog_info!(
            "D3D12.X: depth twin #{} for {res:p} plane {plane}: {}x{}x{} format {} ({} MB in all; footprint formats {} / {}, row pitch {})",
            m.len(),
            d.Width,
            d.Height,
            d.DepthOrArraySize,
            d.Format.0,
            total >> 20,
            fp_twin.Footprint.Format.0,
            fp_plane.Footprint.Format.0,
            fp_plane.Footprint.RowPitch
        );
    }
    Some((info.twin as *mut c_void, plane))
}

pub fn info(res: usize, plane: u32) -> Option<Info> {
    TWINS.lock().unwrap().as_ref()?.get(&(res, plane)).map(|t| t.info)
}

pub fn has(res: usize) -> bool {
    any() && TWINS.lock().unwrap().as_ref().is_some_and(|m| m.contains_key(&(res, 0)) || m.contains_key(&(res, 1)))
}

/// A view of the twin of (`res`, `plane`) is in the console slot whose descriptor starts with `w0`.
pub fn note_view(w0: u32, res: usize, plane: u32) {
    VIEWS.lock().unwrap().get_or_insert_with(HashMap::new).insert(w0, (res, plane));
}

/// The console slot whose image descriptor starts with `w0` is given another view.
pub fn forget_view(w0: u32) {
    if any() {
        if let Some(m) = VIEWS.lock().unwrap().as_mut() {
            m.remove(&w0);
        }
    }
}

/// (depth buffer, plane) whose twin a descriptor starting with `w0` names.
pub fn by_word0(w0: u32) -> Option<(usize, u32)> {
    VIEWS.lock().unwrap().as_ref()?.get(&w0).copied()
}

/// The title's barrier leaves subresource `sub` (u32::MAX: all) of `res` in the PC state `state`.
pub fn set_state(res: usize, sub: u32, state: u32) {
    let mut g = STATES.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if sub == u32::MAX {
        m.retain(|k, _| k.0 != res);
    }
    m.insert((res, sub), state);
}

/// The PC state of subresource `sub` of `res` as the title's barriers left it (DEPTH_WRITE
/// when it never moved it: what `device_x::pc_initial_state` creates these in).
pub fn state(res: usize, sub: u32) -> u32 {
    let g = STATES.lock().unwrap();
    g.as_ref().and_then(|m| m.get(&(res, sub)).or_else(|| m.get(&(res, u32::MAX))).copied()).unwrap_or(0x10)
}
