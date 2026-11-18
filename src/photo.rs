//! Stereo photo pairs: finding `<name>.left.jpg` / `<name>.right.jpg` files,
//! decoding them, working out the camera's field of view from EXIF, and
//! uploading each pair as the two layers of one Vulkan array image, so the
//! shader can pick the layer with `view_index` and show each eye its own photo.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use ash::vk;
use image::{ImageDecoder, ImageReader, imageops::FilterType};

/// Format of the photo texture. The JPEGs hold sRGB-encoded colour.
pub const FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

/// Photos are scaled down to at most this many pixels on their longer side,
/// which roughly matches a current headset's per-eye resolution over the
/// photo's field of view. More would only alias, as there are no mipmaps.
const MAX_SIDE: u32 = 2048;

/// Sensor diagonal assumed when a photo's EXIF has a focal length but no
/// 35 mm equivalent: a 1/2" sensor, as in the Fairphone 4's cameras and many
/// other phones. Photos from a camera with a different sensor will look too
/// large (smaller sensor) or too small (larger sensor).
const ASSUMED_SENSOR_DIAGONAL_MM: f32 = 8.0;
/// Diagonal of a 36 mm x 24 mm frame, the reference for 35 mm equivalents.
const FULL_FRAME_DIAGONAL_MM: f32 = 43.27;
/// Field of view across the long side for photos without a focal length.
const FALLBACK_LONG_SIDE_FOV_DEGREES: f32 = 60.0;

/// Left and right photo files that make up one stereo pair.
#[derive(Debug, PartialEq)]
pub struct PairFiles {
    pub name: String,
    pub left: PathBuf,
    pub right: PathBuf,
}

/// Find the stereo pairs in `dir`, sorted by name: every `<name>.left.jpg`
/// with a matching `<name>.right.jpg` (case-insensitive extension). Halves
/// without a partner are reported on stderr and skipped.
pub fn find_pairs(dir: &Path) -> Result<Vec<PairFiles>> {
    let mut names = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        if let Some(name) = entry?.file_name().to_str() {
            names.push(name.to_owned());
        }
    }
    let (pairs, unpaired) = match_pairs(names);
    for name in unpaired {
        eprintln!("warning: {name} has no matching left/right partner; skipping it");
    }
    Ok(pairs
        .into_iter()
        .map(|(name, left, right)| PairFiles {
            name,
            left: dir.join(left),
            right: dir.join(right),
        })
        .collect())
}

/// Group file names into `(name, left file, right file)` pairs, sorted by
/// name, plus the left/right files that have no partner. Unrelated files are
/// ignored.
fn match_pairs(file_names: Vec<String>) -> (Vec<(String, String, String)>, Vec<String>) {
    // name -> (left file, right file)
    let mut halves: BTreeMap<String, (Option<String>, Option<String>)> = BTreeMap::new();
    for file in file_names {
        let lower = file.to_lowercase();
        for (suffix, is_left) in [(".left.jpg", true), (".right.jpg", false)] {
            if lower.ends_with(suffix) && lower.len() > suffix.len() {
                let name = file[..file.len() - suffix.len()].to_owned();
                let entry = halves.entry(name).or_default();
                if is_left {
                    entry.0 = Some(file.clone());
                } else {
                    entry.1 = Some(file.clone());
                }
            }
        }
    }
    let mut pairs = Vec::new();
    let mut unpaired = Vec::new();
    for (name, halves) in halves {
        match halves {
            (Some(left), Some(right)) => pairs.push((name, left, right)),
            (left, right) => unpaired.extend(left.into_iter().chain(right)),
        }
    }
    (pairs, unpaired)
}

/// Decoded left and right photos, tightly packed RGBA8, left layer first.
pub struct StereoPair {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
    /// Field of view across the longer side, in radians, as captured by the
    /// camera (from the left photo's EXIF).
    pub long_side_fov: f32,
}

impl StereoPair {
    /// A 1x1 black pair, for when there is nothing to show but a texture must
    /// still be bound.
    pub fn placeholder() -> Self {
        Self {
            width: 1,
            height: 1,
            pixels: vec![0, 0, 0, 255, 0, 0, 0, 255],
            long_side_fov: FALLBACK_LONG_SIDE_FOV_DEGREES.to_radians(),
        }
    }

    pub fn aspect(&self) -> f32 {
        self.width as f32 / self.height as f32
    }
}

/// Decode a left/right photo pair, honouring each file's EXIF orientation.
/// Both must end up the same size.
pub fn load_pair(left: &Path, right: &Path) -> Result<StereoPair> {
    let (left, lens) = load_rgba(left)?;
    let (right, _) = load_rgba(right)?;
    if left.dimensions() != right.dimensions() {
        bail!(
            "left and right photos differ in size: {:?} vs {:?}",
            left.dimensions(),
            right.dimensions()
        );
    }
    let (width, height) = left.dimensions();
    let long_side_fov = long_side_fov(lens, width.max(height) as f32 / width.min(height) as f32);
    let mut pixels = left.into_raw();
    pixels.extend_from_slice(right.as_raw());
    Ok(StereoPair {
        width,
        height,
        pixels,
        long_side_fov,
    })
}

/// Lens information from EXIF, in millimetres.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Lens {
    focal_length: Option<f32>,
    focal_length_35mm: Option<f32>,
}

/// Field of view across the long side of a photo with long/short side ratio
/// `ratio`, in radians. Prefers the 35 mm equivalent focal length, which
/// needs no knowledge of the sensor; both equivalents are defined by the
/// diagonal.
fn long_side_fov(lens: Lens, ratio: f32) -> f32 {
    let half_diagonal_tan = match lens {
        Lens {
            focal_length_35mm: Some(f35),
            ..
        } => FULL_FRAME_DIAGONAL_MM / 2.0 / f35,
        Lens {
            focal_length: Some(f),
            ..
        } => ASSUMED_SENSOR_DIAGONAL_MM / 2.0 / f,
        _ => return FALLBACK_LONG_SIDE_FOV_DEGREES.to_radians(),
    };
    // Long side as a fraction of the diagonal.
    let long_fraction = ratio / (ratio * ratio + 1.0).sqrt();
    2.0 * (half_diagonal_tan * long_fraction).atan()
}

fn read_lens(exif: Option<Vec<u8>>) -> Lens {
    let Some(exif) = exif.and_then(|raw| exif::Reader::new().read_raw(raw).ok()) else {
        return Lens::default();
    };
    let get = |tag| {
        let field = exif.get_field(tag, exif::In::PRIMARY)?;
        let value = match &field.value {
            exif::Value::Rational(v) => v.first()?.to_f64() as f32,
            value => value.get_uint(0)? as f32,
        };
        // Some cameras write 0 for "unknown".
        (value.is_finite() && value > 0.0).then_some(value)
    };
    Lens {
        focal_length: get(exif::Tag::FocalLength),
        focal_length_35mm: get(exif::Tag::FocalLengthIn35mmFilm),
    }
}

fn load_rgba(path: &Path) -> Result<(image::RgbaImage, Lens)> {
    let context = || format!("loading {}", path.display());
    let mut decoder = ImageReader::open(path)
        .with_context(context)?
        .with_guessed_format()
        .with_context(context)?
        .into_decoder()
        .with_context(context)?;
    let lens = read_lens(decoder.exif_metadata().with_context(context)?);
    // Phones store pixels in sensor orientation and record the rotation in EXIF.
    let orientation = decoder.orientation().with_context(context)?;
    let mut image = image::DynamicImage::from_decoder(decoder).with_context(context)?;
    image.apply_orientation(orientation);
    if image.width().max(image.height()) > MAX_SIDE {
        image = image.resize(MAX_SIDE, MAX_SIDE, FilterType::Triangle);
    }
    Ok((image.into_rgba8(), lens))
}

/// GPU copy of a stereo pair, ready to be sampled by the fragment shader.
pub struct Texture {
    image: vk::Image,
    memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
}

impl Texture {
    /// Upload `pair` as a two-layer image in SHADER_READ_ONLY_OPTIMAL layout.
    /// Blocks until the copy has finished.
    ///
    /// # Safety
    /// `device` must be a live device created from `instance` and
    /// `physical_device`, and `queue` must belong to the family `cmd_pool` was
    /// created for. The caller must not use `queue` from another thread
    /// during the call.
    pub unsafe fn upload(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        queue: vk::Queue,
        cmd_pool: vk::CommandPool,
        pair: &StereoPair,
    ) -> Result<Self> {
        // SAFETY: forwarded from the caller; every handle created here is
        // either returned or destroyed before returning.
        unsafe {
            let memory_props = instance.get_physical_device_memory_properties(physical_device);
            let size = pair.pixels.len() as vk::DeviceSize;

            // Staging buffer the CPU can write into.
            let staging = device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;
            let staging_reqs = device.get_buffer_memory_requirements(staging);
            let staging_memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(staging_reqs.size)
                    .memory_type_index(find_memory_type(
                        &memory_props,
                        staging_reqs.memory_type_bits,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )?),
                None,
            )?;
            device.bind_buffer_memory(staging, staging_memory, 0)?;
            let mapped = device.map_memory(staging_memory, 0, size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(
                pair.pixels.as_ptr(),
                mapped.cast::<u8>(),
                pair.pixels.len(),
            );
            device.unmap_memory(staging_memory);

            // Device-local array image, one layer per eye.
            let extent = vk::Extent3D {
                width: pair.width,
                height: pair.height,
                depth: 1,
            };
            let image = device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(FORMAT)
                    .extent(extent)
                    .mip_levels(1)
                    .array_layers(2)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?;
            let image_reqs = device.get_image_memory_requirements(image);
            let memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(image_reqs.size)
                    .memory_type_index(find_memory_type(
                        &memory_props,
                        image_reqs.memory_type_bits,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    )?),
                None,
            )?;
            device.bind_image_memory(image, memory, 0)?;

            let layers = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 2,
            };

            // Copy: UNDEFINED -> TRANSFER_DST -> copy -> SHADER_READ_ONLY.
            let cmd = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .command_buffer_count(1),
            )?[0];
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .image(image)
                    .subresource_range(layers)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)],
            );
            // The layers sit back to back in the buffer, so one region covers both.
            device.cmd_copy_buffer_to_image(
                cmd,
                staging,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 2,
                    },
                    image_offset: vk::Offset3D::default(),
                    image_extent: extent,
                }],
            );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .image(image)
                    .subresource_range(layers)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)],
            );
            device.end_command_buffer(cmd)?;
            device.queue_submit(
                queue,
                &[vk::SubmitInfo::default().command_buffers(&[cmd])],
                vk::Fence::null(),
            )?;
            device.queue_wait_idle(queue)?;
            device.free_command_buffers(cmd_pool, &[cmd]);
            device.destroy_buffer(staging, None);
            device.free_memory(staging_memory, None);

            let view = device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
                    .format(FORMAT)
                    .subresource_range(layers),
                None,
            )?;
            let sampler = device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(0.0),
                None,
            )?;

            Ok(Self {
                image,
                memory,
                view,
                sampler,
            })
        }
    }

    /// # Safety
    /// The GPU must have finished every command that uses this texture.
    pub unsafe fn destroy(self, device: &ash::Device) {
        // SAFETY: forwarded from the caller.
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Result<u32> {
    (0..props.memory_type_count)
        .find(|&i| {
            type_bits & (1 << i) != 0
                && props.memory_types[i as usize]
                    .property_flags
                    .contains(flags)
        })
        .with_context(|| format!("no Vulkan memory type with {flags:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn images() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/images")
    }

    #[test]
    fn exif_rotated_pair_loads_upright() {
        let dir = images();
        let pair = load_pair(&dir.join("lamp.left.jpg"), &dir.join("lamp.right.jpg")).unwrap();
        // The files are stored 4000x3000 with an EXIF rotation to portrait.
        assert!(pair.height > pair.width, "EXIF orientation not applied");
        assert_eq!(pair.width.max(pair.height), MAX_SIDE);
        assert_eq!(
            pair.pixels.len(),
            2 * 4 * (pair.width * pair.height) as usize
        );
        // Fairphone 4 main camera, 5.6 mm on a 6.4 mm wide sensor (rounded,
        // hence the 1 degree tolerance).
        let expected = 2.0 * (3.2f32 / 5.6).atan();
        assert!((pair.long_side_fov - expected).abs() < 1f32.to_radians());
    }

    #[test]
    fn landscape_pair_loads() {
        let dir = images();
        let pair = load_pair(&dir.join("table.left.jpg"), &dir.join("table.right.jpg")).unwrap();
        assert_eq!((pair.width, pair.height), (MAX_SIDE, MAX_SIDE * 3 / 4));
        // Fairphone 4 wide camera, 3.6 mm on a 6.4 mm wide sensor (rounded,
        // hence the 1 degree tolerance).
        let expected = 2.0 * (3.2f32 / 3.6).atan();
        assert!((pair.long_side_fov - expected).abs() < 1f32.to_radians());
    }

    #[test]
    fn bundled_pairs_are_found() {
        let names: Vec<_> = find_pairs(&images())
            .unwrap()
            .into_iter()
            .map(|p| p.name)
            .collect();
        assert_eq!(names, ["lamp", "table"]);
    }

    #[test]
    fn pairs_need_both_halves() {
        let files = [
            "b.left.jpg",
            "b.right.JPG",
            "a.left.jpg",
            "a.right.jpg",
            "lonely.left.jpg",
            "notes.txt",
            "plain.jpg",
            ".left.jpg",
        ];
        let (pairs, unpaired) = match_pairs(files.map(String::from).to_vec());
        let names: Vec<_> = pairs.iter().map(|p| p.0.as_str()).collect();
        assert_eq!(names, ["a", "b"]);
        assert_eq!(pairs[1].2, "b.right.JPG");
        assert_eq!(unpaired, ["lonely.left.jpg"]);
    }

    #[test]
    fn fov_prefers_35mm_equivalent() {
        let lens = Lens {
            focal_length: Some(1.0),
            focal_length_35mm: Some(24.0),
        };
        // 3:2 like full frame, so the long side is 36 mm.
        let expected = 2.0 * (18.0f32 / 24.0).atan();
        assert!((long_side_fov(lens, 1.5) - expected).abs() < 1e-3);
        assert_eq!(
            long_side_fov(Lens::default(), 1.5),
            FALLBACK_LONG_SIDE_FOV_DEGREES.to_radians()
        );
    }
}
