//! Minimal OpenXR + Vulkan "hello world" for SteamVR, written against the
//! `openxr` and `ash` crates.
//!
//! It renders a rotating cube floating above a floor quad, plus a floating
//! button that can be pointed at with either controller's laser and clicked
//! with the trigger to pause or resume the cube. Both eyes are drawn
//! in a single pass using Vulkan multiview. The shader lives in
//! `src/shaders/scene.wgsl` and is compiled to SPIR-V at startup with naga, so
//! no external shader compiler is required.

use std::{
    env,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use ash::vk::{self, Handle};
use glam::{Mat4, Quat, Vec3, Vec4};
use openxr as xr;

const COLOR_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;
const VIEW_COUNT: u32 = 2;
const VIEW_TYPE: xr::ViewConfigurationType = xr::ViewConfigurationType::PRIMARY_STEREO;
/// Number of frames that may be in flight on the GPU at once.
const PIPELINE_DEPTH: u32 = 2;
const NEAR: f32 = 0.05;
const FAR: f32 = 100.0;
const CUBE_VERTS: u32 = 36;
const FLOOR_VERTS: u32 = 6;
const TEXT_VERTS: u32 = 6;

// Colour styles for cube-geometry draws, passed as the first instance and
// matching STYLE in the shader.
const STYLE_NORMALS: u32 = 0;
const STYLE_BUTTON_IDLE: u32 = 1;
const STYLE_BUTTON_HOVER: u32 = 2;
const STYLE_BUTTON_PRESSED: u32 = 3;
const STYLE_LASER: u32 = 4;
const STYLE_CONTROLLER: u32 = 5;

/// Centre of the floating button in STAGE space: in front of and below the cube.
const BUTTON_CENTER: Vec3 = Vec3::new(0.0, 1.0, -1.0);
/// Half extents of the button box; it faces +Z, toward the user.
const BUTTON_HALF_EXTENTS: Vec3 = Vec3::new(0.1, 0.1, 0.03);
/// How far the button face sinks while it is held down.
const BUTTON_PRESS_DEPTH: f32 = 0.02;
/// Length of the laser when it does not hit the button.
const LASER_LENGTH: f32 = 3.0;
const LASER_RADIUS: f32 = 0.002;

const SHADER_SOURCE: &str = include_str!("shaders/scene.wgsl");

/// Push constant block matching `Push` in the shader: one MVP per eye.
#[repr(C)]
#[derive(Clone, Copy)]
struct PushConstants {
    mvp: [[f32; 16]; VIEW_COUNT as usize],
}

impl PushConstants {
    fn as_bytes(&self) -> &[u8] {
        // SAFETY: PushConstants is repr(C) and made only of f32, so any byte
        // pattern is a valid representation and there is no padding.
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }
}

struct Swapchain {
    handle: xr::Swapchain<xr::Vulkan>,
    buffers: Vec<Framebuffer>,
    resolution: vk::Extent2D,
}

struct Framebuffer {
    framebuffer: vk::Framebuffer,
    color: vk::ImageView,
}

/// Controller input: a pointing pose, a trigger and haptics for each hand.
struct Input {
    action_set: xr::ActionSet,
    select: xr::Action<bool>,
    haptic: xr::Action<xr::Haptic>,
    // Kept alive for the spaces created from it.
    _aim: xr::Action<xr::Posef>,
    hands: [Hand; 2],
}

struct Hand {
    path: xr::Path,
    aim_space: xr::Space,
}

fn main() -> Result<()> {
    let running = Arc::new(AtomicBool::new(true));
    {
        let running = running.clone();
        ctrlc::set_handler(move || running.store(false, Ordering::Relaxed))
            .context("installing Ctrl-C handler")?;
    }

    let entry = load_entry()?;

    let available = entry.enumerate_extensions()?;
    if !available.khr_vulkan_enable2 {
        bail!("the active OpenXR runtime does not support XR_KHR_vulkan_enable2");
    }
    let mut extensions = xr::ExtensionSet::default();
    extensions.khr_vulkan_enable2 = true;

    let xr_instance = entry
        .create_instance(
            &xr::ApplicationInfo {
                application_name: "hello-vr",
                application_version: 0,
                engine_name: "hello-vr",
                engine_version: 0,
                api_version: xr::Version::new(1, 0, 0),
            },
            &extensions,
            &[],
            &(),
        )
        .map_err(|e| match e {
            xr::sys::Result::ERROR_RUNTIME_FAILURE => anyhow!(
                "the OpenXR runtime refused to create an instance ({e}). With SteamVR this \
             usually means no headset is connected; check SteamVR's status window and \
             ~/.steam/steam/logs/vrserver.txt for 'Refusing connect'"
            ),
            e => anyhow!("xrCreateInstance failed: {e}"),
        })?;
    let props = xr_instance.properties()?;
    println!(
        "OpenXR runtime: {} {}",
        props.runtime_name, props.runtime_version
    );

    let system = xr_instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)?;
    let blend_mode = xr_instance.enumerate_environment_blend_modes(system, VIEW_TYPE)?[0];

    // Vulkan 1.1 guarantees multiview.
    let vk_target_version = vk::make_api_version(0, 1, 1, 0);
    let vk_target_version_xr = xr::Version::new(1, 1, 0);
    let reqs = xr_instance.graphics_requirements::<xr::Vulkan>(system)?;
    if vk_target_version_xr < reqs.min_api_version_supported
        || vk_target_version_xr.major() > reqs.max_api_version_supported.major()
    {
        bail!(
            "runtime requires Vulkan between {} and {}.x",
            reqs.min_api_version_supported,
            reqs.max_api_version_supported.major()
        );
    }

    let vert_spv = compile_wgsl(SHADER_SOURCE, naga::ShaderStage::Vertex, "vs_main")?;
    let frag_spv = compile_wgsl(SHADER_SOURCE, naga::ShaderStage::Fragment, "fs_main")?;

    // SAFETY: everything below is raw Vulkan/OpenXR FFI. Handles are created
    // in dependency order and destroyed in reverse at the end of this block.
    unsafe {
        let vk_entry = ash::Entry::load().context("loading libvulkan")?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"hello-vr")
            .api_version(vk_target_version);

        // The runtime creates the Vulkan instance and device for us so it can
        // inject whatever extensions it needs.
        let get_instance_proc_addr: xr::sys::platform::VkGetInstanceProcAddr =
            std::mem::transmute(vk_entry.static_fn().get_instance_proc_addr);

        let vk_instance = {
            let raw = xr_instance
                .create_vulkan_instance(
                    system,
                    get_instance_proc_addr,
                    &vk::InstanceCreateInfo::default().application_info(&app_info) as *const _
                        as *const _,
                )
                .context("xrCreateVulkanInstanceKHR")?
                .map_err(vk::Result::from_raw)
                .context("vkCreateInstance")?;
            ash::Instance::load(vk_entry.static_fn(), vk::Instance::from_raw(raw as _))
        };

        let physical_device = vk::PhysicalDevice::from_raw(
            xr_instance.vulkan_graphics_device(system, vk_instance.handle().as_raw() as _)? as _,
        );
        let device_props = vk_instance.get_physical_device_properties(physical_device);
        println!(
            "Vulkan device: {}",
            device_props
                .device_name_as_c_str()
                .unwrap_or(c"?")
                .to_string_lossy()
        );
        if device_props.api_version < vk_target_version {
            bail!("Vulkan physical device does not support Vulkan 1.1");
        }

        let queue_family_index = vk_instance
            .get_physical_device_queue_family_properties(physical_device)
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .context("no graphics queue family")? as u32;

        let device = {
            let queue_info = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&[1.0])];
            let mut multiview = vk::PhysicalDeviceMultiviewFeatures::default().multiview(true);
            let info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_info)
                .push_next(&mut multiview);
            let raw = xr_instance
                .create_vulkan_device(
                    system,
                    get_instance_proc_addr,
                    physical_device.as_raw() as _,
                    &info as *const _ as *const _,
                )
                .context("xrCreateVulkanDeviceKHR")?
                .map_err(vk::Result::from_raw)
                .context("vkCreateDevice")?;
            ash::Device::load(vk_instance.fp_v1_0(), vk::Device::from_raw(raw as _))
        };
        let queue = device.get_device_queue(queue_family_index, 0);

        // One render pass that writes both array layers via multiview.
        let view_mask = (1u32 << VIEW_COUNT) - 1;
        let render_pass = {
            let attachments = [vk::AttachmentDescription {
                format: COLOR_FORMAT,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                ..Default::default()
            }];
            let color_ref = [vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }];
            let subpasses = [vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&color_ref)];
            let dependencies = [vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                ..Default::default()
            }];
            let view_masks = [view_mask];
            let mut multiview = vk::RenderPassMultiviewCreateInfo::default()
                .view_masks(&view_masks)
                .correlation_masks(&view_masks);
            device.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&attachments)
                    .subpasses(&subpasses)
                    .dependencies(&dependencies)
                    .push_next(&mut multiview),
                None,
            )?
        };

        let vert = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vert_spv), None)?;
        let frag = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&frag_spv), None)?;

        let push_range = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX,
            offset: 0,
            size: std::mem::size_of::<PushConstants>() as u32,
        }];
        let pipeline_layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_range),
            None,
        )?;

        let pipeline = {
            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vert)
                    .name(c"vs_main"),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(frag)
                    .name(c"fs_main"),
            ];
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
            let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);
            // Geometry is wound counter-clockwise when seen from outside in a
            // y-up world. The projection flips y for Vulkan's y-down clip
            // space, so the image on screen matches OpenGL's and the triangles
            // still appear counter-clockwise. Vulkan judges orientation as it
            // appears on screen, so front faces are COUNTER_CLOCKWISE. Culling
            // back faces is enough to draw a convex cube without a depth buffer.
            let raster = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0);
            let multisample = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)];
            let blend =
                vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic =
                vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
            let info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&input_assembly)
                .viewport_state(&viewport)
                .rasterization_state(&raster)
                .multisample_state(&multisample)
                .color_blend_state(&blend)
                .dynamic_state(&dynamic)
                .layout(pipeline_layout)
                .render_pass(render_pass)
                .subpass(0);
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
                .map_err(|(_, e)| e)?[0]
        };
        device.destroy_shader_module(vert, None);
        device.destroy_shader_module(frag, None);

        let (session, mut frame_wait, mut frame_stream) = xr_instance
            .create_session::<xr::Vulkan>(
                system,
                &xr::vulkan::SessionCreateInfo {
                    instance: vk_instance.handle().as_raw() as _,
                    physical_device: physical_device.as_raw() as _,
                    device: device.handle().as_raw() as _,
                    queue_family_index,
                    queue_index: 0,
                },
            )
            .context("creating OpenXR session")?;

        // STAGE space puts the origin on the floor at the centre of the play area.
        let stage =
            session.create_reference_space(xr::ReferenceSpaceType::STAGE, xr::Posef::IDENTITY)?;
        let input = create_input(&xr_instance, &session)?;

        let cmd_pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family_index)
                .flags(
                    vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                        | vk::CommandPoolCreateFlags::TRANSIENT,
                ),
            None,
        )?;
        let cmds = device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .command_buffer_count(PIPELINE_DEPTH),
        )?;
        let fences = (0..PIPELINE_DEPTH)
            .map(|_| {
                device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let mut swapchain: Option<Swapchain> = None;
        let mut event_storage = xr::EventDataBuffer::new();
        let mut session_running = false;
        let mut frame = 0usize;
        let mut last_frame = Instant::now();
        let mut spin_angle = 0.0f32;
        let mut spinning = true;

        println!("entering main loop; press Ctrl-C to exit");
        'main: loop {
            if !running.load(Ordering::Relaxed) {
                match session.request_exit() {
                    Ok(()) => {}
                    Err(xr::sys::Result::ERROR_SESSION_NOT_RUNNING) => break,
                    Err(e) => return Err(e.into()),
                }
            }

            while let Some(event) = xr_instance.poll_event(&mut event_storage)? {
                use xr::Event::*;
                match event {
                    SessionStateChanged(e) => {
                        println!("session state: {:?}", e.state());
                        match e.state() {
                            xr::SessionState::READY => {
                                session.begin(VIEW_TYPE)?;
                                session_running = true;
                            }
                            xr::SessionState::STOPPING => {
                                session.end()?;
                                session_running = false;
                            }
                            xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => {
                                break 'main;
                            }
                            _ => {}
                        }
                    }
                    InstanceLossPending(_) => break 'main,
                    EventsLost(e) => println!("lost {} events", e.lost_event_count()),
                    _ => {}
                }
            }

            if !session_running {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }

            let frame_state = frame_wait.wait()?;
            frame_stream.begin()?;
            // Returns XR_SESSION_NOT_FOCUSED (a success code) while another app
            // has input focus; the actions then simply report inactive.
            session.sync_actions(&[xr::ActiveActionSet::new(&input.action_set)])?;

            if !frame_state.should_render {
                frame_stream.end(frame_state.predicted_display_time, blend_mode, &[])?;
                continue;
            }

            let swapchain = match &mut swapchain {
                Some(s) => s,
                None => swapchain.insert(create_swapchain(
                    &xr_instance,
                    system,
                    &session,
                    &device,
                    render_pass,
                )?),
            };

            let image_index = swapchain.handle.acquire_image()?;

            let (_, views) =
                session.locate_views(VIEW_TYPE, frame_state.predicted_display_time, &stage)?;
            let view_proj: [Mat4; VIEW_COUNT as usize] = [
                projection(views[0].fov) * view_matrix(views[0].pose),
                projection(views[1].fov) * view_matrix(views[1].pose),
            ];

            // Point each controller's aim ray at the button. A trigger press
            // that starts while the ray is on the button counts as a click.
            let button_min = BUTTON_CENTER - BUTTON_HALF_EXTENTS;
            let button_max = BUTTON_CENTER + BUTTON_HALF_EXTENTS;
            let mut hovered = false;
            let mut held = false;
            let mut pointers: Vec<(Mat4, f32)> = Vec::new();
            for hand in &input.hands {
                let location = hand
                    .aim_space
                    .locate(&stage, frame_state.predicted_display_time)?;
                let valid = xr::SpaceLocationFlags::POSITION_VALID
                    | xr::SpaceLocationFlags::ORIENTATION_VALID;
                if !location.location_flags.contains(valid) {
                    continue;
                }
                let aim = pose_matrix(location.pose);
                let origin = aim.transform_point3(Vec3::ZERO);
                let dir = aim.transform_vector3(Vec3::NEG_Z);
                let hit = ray_box_hit(origin, dir, button_min, button_max);
                pointers.push((aim, hit.unwrap_or(LASER_LENGTH)));

                let trigger = input.select.state(&session, hand.path)?;
                if hit.is_some() {
                    hovered = true;
                    held |= trigger.current_state;
                    if trigger.current_state && trigger.changed_since_last_sync {
                        spinning = !spinning;
                        input.haptic.apply_feedback(
                            &session,
                            hand.path,
                            &xr::HapticVibration::new()
                                .amplitude(0.8)
                                .duration(xr::Duration::from_nanos(50_000_000))
                                .frequency(xr::FREQUENCY_UNSPECIFIED),
                        )?;
                    }
                }
            }

            let now = Instant::now();
            if spinning {
                spin_angle += (now - last_frame).as_secs_f32();
            }
            last_frame = now;

            let push = |model: Mat4| PushConstants {
                mvp: view_proj.map(|vp| (vp * model).to_cols_array()),
            };
            let cube_push = push(
                Mat4::from_translation(Vec3::new(0.0, 1.5, -1.5))
                    * Mat4::from_rotation_y(spin_angle)
                    * Mat4::from_rotation_x(spin_angle * 0.7)
                    * Mat4::from_scale(Vec3::splat(0.2)),
            );
            let floor_push = push(Mat4::IDENTITY);
            // Text hangs above the cube, facing the user at the origin.
            let text_push = push(Mat4::from_translation(Vec3::new(0.0, 2.05, -1.5)));
            // While held, the button's front face sinks toward its back face.
            let press = if held { BUTTON_PRESS_DEPTH } else { 0.0 };
            let button_push = push(
                Mat4::from_translation(BUTTON_CENTER - Vec3::Z * (press / 2.0))
                    * Mat4::from_scale(BUTTON_HALF_EXTENTS - Vec3::Z * (press / 2.0)),
            );
            let button_style = match (hovered, held) {
                (_, true) => STYLE_BUTTON_PRESSED,
                (true, false) => STYLE_BUTTON_HOVER,
                (false, false) => STYLE_BUTTON_IDLE,
            };
            // A small box for each controller body and a thin one for its
            // laser, which stops where it hits the button.
            let pointer_pushes: Vec<(PushConstants, u32)> = pointers
                .iter()
                .flat_map(|&(aim, length)| {
                    [
                        (
                            push(
                                aim * Mat4::from_translation(Vec3::new(0.0, 0.0, 0.05))
                                    * Mat4::from_scale(Vec3::new(0.02, 0.02, 0.05)),
                            ),
                            STYLE_CONTROLLER,
                        ),
                        (
                            push(
                                aim * Mat4::from_translation(Vec3::new(0.0, 0.0, -length / 2.0))
                                    * Mat4::from_scale(Vec3::new(
                                        LASER_RADIUS,
                                        LASER_RADIUS,
                                        length / 2.0,
                                    )),
                            ),
                            STYLE_LASER,
                        ),
                    ]
                })
                .collect();

            device.wait_for_fences(&[fences[frame]], true, u64::MAX)?;
            device.reset_fences(&[fences[frame]])?;

            let cmd = cmds[frame];
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            let clear = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.05, 0.05, 0.1, 1.0],
                },
            }];
            let extent = swapchain.resolution;
            device.cmd_begin_render_pass(
                cmd,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(render_pass)
                    .framebuffer(swapchain.buffers[image_index as usize].framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent,
                    })
                    .clear_values(&clear),
                vk::SubpassContents::INLINE,
            );
            device.cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            device.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent,
                }],
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);

            // The first instance carries the colour style (see STYLE in the shader).
            let draw = |push: &PushConstants, first_vertex: u32, vertex_count: u32, style: u32| {
                device.cmd_push_constants(
                    cmd,
                    pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    push.as_bytes(),
                );
                device.cmd_draw(cmd, vertex_count, 1, first_vertex, style);
            };
            // Draw order stands in for a depth buffer: floor, cube, button (in
            // front of the cube), text, and finally the controllers and lasers,
            // which are always closest to the user.
            draw(&floor_push, CUBE_VERTS, FLOOR_VERTS, STYLE_NORMALS);
            draw(&cube_push, 0, CUBE_VERTS, STYLE_NORMALS);
            draw(&button_push, 0, CUBE_VERTS, button_style);
            draw(
                &text_push,
                CUBE_VERTS + FLOOR_VERTS,
                TEXT_VERTS,
                STYLE_NORMALS,
            );
            for (push, style) in &pointer_pushes {
                draw(push, 0, CUBE_VERTS, *style);
            }

            device.cmd_end_render_pass(cmd);
            device.end_command_buffer(cmd)?;

            // The compositor may still be reading the image; wait before we write.
            swapchain.handle.wait_image(xr::Duration::INFINITE)?;
            device.queue_submit(
                queue,
                &[vk::SubmitInfo::default().command_buffers(&[cmd])],
                fences[frame],
            )?;
            swapchain.handle.release_image()?;

            let rect = xr::Rect2Di {
                offset: xr::Offset2Di { x: 0, y: 0 },
                extent: xr::Extent2Di {
                    width: extent.width as i32,
                    height: extent.height as i32,
                },
            };
            let projection_views = [
                xr::CompositionLayerProjectionView::new()
                    .pose(views[0].pose)
                    .fov(views[0].fov)
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&swapchain.handle)
                            .image_array_index(0)
                            .image_rect(rect),
                    ),
                xr::CompositionLayerProjectionView::new()
                    .pose(views[1].pose)
                    .fov(views[1].fov)
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&swapchain.handle)
                            .image_array_index(1)
                            .image_rect(rect),
                    ),
            ];
            frame_stream.end(
                frame_state.predicted_display_time,
                blend_mode,
                &[&xr::CompositionLayerProjection::new()
                    .space(&stage)
                    .views(&projection_views)],
            )?;
            frame = (frame + 1) % PIPELINE_DEPTH as usize;
        }

        // OpenXR objects must go before the Vulkan objects they reference.
        drop((input, session, frame_wait, frame_stream, stage));

        device.wait_for_fences(&fences, true, u64::MAX)?;
        for fence in fences {
            device.destroy_fence(fence, None);
        }
        if let Some(swapchain) = swapchain {
            for buffer in swapchain.buffers {
                device.destroy_framebuffer(buffer.framebuffer, None);
                device.destroy_image_view(buffer.color, None);
            }
            drop(swapchain.handle);
        }
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_command_pool(cmd_pool, None);
        device.destroy_render_pass(render_pass, None);
        device.destroy_device(None);
        vk_instance.destroy_instance(None);
    }

    println!("exited cleanly");
    Ok(())
}

/// Locate the OpenXR loader. Checks `OPENXR_LOADER`, then the system library
/// search path, then the copy that ships with SteamVR.
fn load_entry() -> Result<xr::Entry> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(path) = env::var("OPENXR_LOADER") {
        candidates.push(path.into());
    }
    candidates.push("libopenxr_loader.so.1".into());
    candidates.push("libopenxr_loader.so".into());
    if let Ok(home) = env::var("HOME") {
        for steam in [".steam/steam", ".local/share/Steam"] {
            candidates.push(
                Path::new(&home)
                    .join(steam)
                    .join("steamapps/common/SteamVR/bin/linux64/libopenxr_loader.so"),
            );
        }
    }
    for path in &candidates {
        // SAFETY: we only ever point this at an actual OpenXR loader library.
        if let Ok(entry) = unsafe { xr::Entry::load_from(path, &()) } {
            println!("OpenXR loader: {}", path.display());
            return Ok(entry);
        }
    }
    bail!(
        "could not find libopenxr_loader.so; tried {:?}. Set OPENXR_LOADER to its path.",
        candidates
    )
}

/// Compile one entry point of the WGSL source to SPIR-V 1.3.
fn compile_wgsl(source: &str, stage: naga::ShaderStage, entry_point: &str) -> Result<Vec<u32>> {
    use naga::back::spv;

    let module = naga::front::wgsl::parse_str(source)
        .map_err(|e| anyhow!("WGSL parse error:\n{}", e.emit_to_string(source)))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| anyhow!("WGSL validation error:\n{}", e.emit_to_string(source)))?;

    let options = spv::Options {
        lang_version: (1, 3),
        // We build Vulkan-convention projection matrices ourselves, so don't
        // let naga flip y or remap z.
        flags: spv::WriterFlags::empty(),
        ..Default::default()
    };
    let pipeline_options = spv::PipelineOptions {
        shader_stage: stage,
        entry_point: entry_point.to_owned(),
    };
    spv::write_vec(&module, &info, &options, Some(&pipeline_options))
        .with_context(|| format!("emitting SPIR-V for {entry_point}"))
}

unsafe fn create_swapchain(
    xr_instance: &xr::Instance,
    system: xr::SystemId,
    session: &xr::Session<xr::Vulkan>,
    device: &ash::Device,
    render_pass: vk::RenderPass,
) -> Result<Swapchain> {
    let views = xr_instance.enumerate_view_configuration_views(system, VIEW_TYPE)?;
    if views.len() != VIEW_COUNT as usize {
        bail!(
            "expected {VIEW_COUNT} views, runtime reports {}",
            views.len()
        );
    }
    // Multiview renders both eyes into one array image, so they must match.
    if views[0] != views[1] {
        bail!("per-eye view configurations differ; multiview needs identical sizes");
    }
    let resolution = vk::Extent2D {
        width: views[0].recommended_image_rect_width,
        height: views[0].recommended_image_rect_height,
    };
    println!(
        "swapchain: {}x{} per eye",
        resolution.width, resolution.height
    );

    let handle = session.create_swapchain(&xr::SwapchainCreateInfo {
        create_flags: xr::SwapchainCreateFlags::EMPTY,
        usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT | xr::SwapchainUsageFlags::SAMPLED,
        format: COLOR_FORMAT.as_raw() as _,
        sample_count: 1,
        width: resolution.width,
        height: resolution.height,
        face_count: 1,
        array_size: VIEW_COUNT,
        mip_count: 1,
    })?;

    let buffers = handle
        .enumerate_images()?
        .into_iter()
        .map(|raw| {
            let image = vk::Image::from_raw(raw);
            // SAFETY: caller guarantees `device` and `render_pass` are live.
            unsafe {
                let color = device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
                        .format(COLOR_FORMAT)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: VIEW_COUNT,
                        }),
                    None,
                )?;
                let attachments = [color];
                let framebuffer = device.create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(render_pass)
                        .width(resolution.width)
                        .height(resolution.height)
                        .attachments(&attachments)
                        .layers(1),
                    None,
                )?;
                Ok(Framebuffer { framebuffer, color })
            }
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(Swapchain {
        handle,
        buffers,
        resolution,
    })
}

/// Create the controller actions, suggest bindings for common controllers and
/// attach them to the session. Must run before the session's first sync.
fn create_input(instance: &xr::Instance, session: &xr::Session<xr::Vulkan>) -> Result<Input> {
    let path = |s: &str| {
        instance
            .string_to_path(s)
            .with_context(|| format!("xrStringToPath({s})"))
    };
    let hand_paths = [path("/user/hand/left")?, path("/user/hand/right")?];

    let action_set = instance.create_action_set("gameplay", "Gameplay", 0)?;
    let aim = action_set.create_action::<xr::Posef>("aim", "Aim", &hand_paths)?;
    let select = action_set.create_action::<bool>("select", "Select", &hand_paths)?;
    let haptic = action_set.create_action::<xr::Haptic>("haptic", "Haptic", &hand_paths)?;

    // Each profile names the input that acts as the trigger. Float inputs
    // such as trigger/value are converted to a boolean by the runtime.
    let profiles = [
        (
            "/interaction_profiles/khr/simple_controller",
            "select/click",
        ),
        (
            "/interaction_profiles/valve/index_controller",
            "trigger/click",
        ),
        ("/interaction_profiles/htc/vive_controller", "trigger/click"),
        (
            "/interaction_profiles/oculus/touch_controller",
            "trigger/value",
        ),
    ];
    for (profile, trigger) in profiles {
        let mut bindings = Vec::new();
        for hand in ["left", "right"] {
            let prefix = format!("/user/hand/{hand}");
            bindings.push(xr::Binding::new(
                &aim,
                path(&format!("{prefix}/input/aim/pose"))?,
            ));
            bindings.push(xr::Binding::new(
                &select,
                path(&format!("{prefix}/input/{trigger}"))?,
            ));
            bindings.push(xr::Binding::new(
                &haptic,
                path(&format!("{prefix}/output/haptic"))?,
            ));
        }
        // One unsupported profile should not stop the others from working.
        if let Err(e) = instance.suggest_interaction_profile_bindings(path(profile)?, &bindings) {
            eprintln!("warning: could not suggest bindings for {profile}: {e}");
        }
    }

    session.attach_action_sets(&[&action_set])?;

    let hand = |path| -> Result<Hand> {
        Ok(Hand {
            path,
            aim_space: aim.create_space(session, path, xr::Posef::IDENTITY)?,
        })
    };
    let hands = [hand(hand_paths[0])?, hand(hand_paths[1])?];
    Ok(Input {
        action_set,
        select,
        haptic,
        _aim: aim,
        hands,
    })
}

/// Distance along a ray to where it first touches an axis-aligned box, or
/// `None` if it misses or the box is entirely behind the origin. An origin
/// inside the box gives 0.
fn ray_box_hit(origin: Vec3, dir: Vec3, min: Vec3, max: Vec3) -> Option<f32> {
    // Slab test. A zero direction component gives infinite slab distances,
    // which accept or reject that axis (barring an origin exactly on a face).
    let inv = dir.recip();
    let t1 = (min - origin) * inv;
    let t2 = (max - origin) * inv;
    let enter = t1.min(t2).max_element();
    let exit = t1.max(t2).min_element();
    (enter <= exit && exit >= 0.0).then_some(enter.max(0.0))
}

/// Local-to-world transform for an OpenXR pose.
fn pose_matrix(pose: xr::Posef) -> Mat4 {
    let rotation = Quat::from_xyzw(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    );
    let position = Vec3::new(pose.position.x, pose.position.y, pose.position.z);
    Mat4::from_rotation_translation(rotation, position)
}

/// World-to-eye transform for an OpenXR view pose.
fn view_matrix(pose: xr::Posef) -> Mat4 {
    pose_matrix(pose).inverse()
}

/// Asymmetric off-axis projection for an OpenXR field of view, using Vulkan
/// clip-space conventions (y down, depth in [0, 1]). Mirrors
/// `XrMatrix4x4f_CreateProjectionFov` from the OpenXR SDK's xr_linear.h.
fn projection(fov: xr::Fovf) -> Mat4 {
    let left = fov.angle_left.tan();
    let right = fov.angle_right.tan();
    let up = fov.angle_up.tan();
    let down = fov.angle_down.tan();
    let width = right - left;
    let height = down - up; // flipped for Vulkan's y-down clip space
    Mat4::from_cols(
        Vec4::new(2.0 / width, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 / height, 0.0, 0.0),
        Vec4::new(
            (right + left) / width,
            (up + down) / height,
            -FAR / (FAR - NEAR),
            -1.0,
        ),
        Vec4::new(0.0, 0.0, -(FAR * NEAR) / (FAR - NEAR), 0.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaders_compile() {
        let vert = compile_wgsl(SHADER_SOURCE, naga::ShaderStage::Vertex, "vs_main").unwrap();
        let frag = compile_wgsl(SHADER_SOURCE, naga::ShaderStage::Fragment, "fs_main").unwrap();
        assert!(!vert.is_empty() && !frag.is_empty());
    }

    #[test]
    fn push_constants_fit_vulkan_minimum() {
        assert!(std::mem::size_of::<PushConstants>() <= 128);
    }

    #[test]
    fn ray_box_hit_reports_entry_distance() {
        let (min, max) = (Vec3::splat(-1.0), Vec3::splat(1.0));
        // Straight at the box from 5 m away along +Z.
        let hit = ray_box_hit(Vec3::new(0.0, 0.0, 5.0), Vec3::NEG_Z, min, max);
        assert!((hit.unwrap() - 4.0).abs() < 1e-5);
        // Pointing away from it.
        assert_eq!(
            ray_box_hit(Vec3::new(0.0, 0.0, 5.0), Vec3::Z, min, max),
            None
        );
        // Passing beside it, parallel to an axis.
        assert_eq!(
            ray_box_hit(Vec3::new(2.0, 0.0, 5.0), Vec3::NEG_Z, min, max),
            None
        );
        // Starting inside.
        assert_eq!(ray_box_hit(Vec3::ZERO, Vec3::X, min, max), Some(0.0));
        // At an angle.
        let dir = Vec3::new(-1.0, 0.0, -1.0).normalize();
        let hit = ray_box_hit(Vec3::new(3.0, 0.0, 3.0), dir, min, max).unwrap();
        assert!((hit - 2.0 * 2f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn projection_maps_near_and_far_to_vulkan_depth() {
        let fov = xr::Fovf {
            angle_left: -0.8,
            angle_right: 0.7,
            angle_up: 0.75,
            angle_down: -0.8,
        };
        let p = projection(fov);
        let near = p * Vec4::new(0.0, 0.0, -NEAR, 1.0);
        let far = p * Vec4::new(0.0, 0.0, -FAR, 1.0);
        assert!((near.z / near.w).abs() < 1e-5);
        assert!((far.z / far.w - 1.0).abs() < 1e-4);
        // Something above eye level must end up with negative (upward) clip y.
        let above = p * Vec4::new(0.0, 1.0, -2.0, 1.0);
        assert!(above.y / above.w < 0.0);
    }
}
