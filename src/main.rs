use std::{
    ffi::CString,
    sync::{Arc, atomic::AtomicBool},
};

use clap::Parser;
use sdl3::{
    self,
    image::LoadTexture,
    render::{FPoint, Vertex},
};

#[allow(dead_code)]
mod ffmpeg;

#[derive(clap::Parser)]
#[command(author = "timelessnesses", about = "sdl3 (Rust) Bunnymark")]
pub struct Cli {
    /// List GPU renderers (for the SELECTED_GPU_RENDERER arg)
    #[arg(short, long)]
    pub list_gpu_renderers: bool,

    /// Select your own renderer if you want to
    #[arg(short, long)]
    pub selected_gpu_renderer: Option<u32>,

    /// Width of the window (default: 1280)
    #[arg(short, long, default_value_t = 1280)]
    pub width: u32,

    /// Height of the window (default: 720)
    #[arg(long, default_value_t = 720)]
    pub height: u32,

    /// Texture limit number (default: 50000)
    #[arg(long, default_value_t = 50000)]
    pub texture_limit: u32,

    /// Custom 2D texture folder (default: None (program's included textures))
    #[arg(long)]
    pub texture_folder: Option<std::path::PathBuf>,

    /// Texture width, the aspect ratio will be respected, if this value was set, texture_height will be ignored (default: 32)
    #[arg(long, default_value_t = 32, conflicts_with = "texture_height")]
    pub texture_width: u32,

    /// Texture height, the aspect ratio will be respected if this value was set, texture_width will be ignored
    #[arg(long, default_value_t = 0, conflicts_with = "texture_width")]
    pub texture_height: u32,

    /// Texture time to spawn (default: 0.001 seconds)
    #[arg(long, default_value_t = 0.001)]
    pub texture_spawn_time: f32,

    /// Spawn textures by mouse clicks instead of time (default: false)
    #[arg(long, default_value_t = false)]
    pub spawn_by_click: bool,

    /// Mouse click spawn amount (left click: 100, right click: 10)
    #[arg(long, default_value_t = ClickRange(100, 10), value_parser = convert_mouse_click_string_to_left_right)]
    pub mouse_click_spawn_amount: ClickRange,

    #[arg(long, default_value_t = false)]
    pub record: bool,

    /// Draw textures as geometry instead of copying textures (support only one texture)
    #[arg(long, default_value_t = false)]
    draw_as_geometry: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClickRange(u32, u32);

impl std::fmt::Display for ClickRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{},{}", self.0, self.1)
    }
}

fn convert_mouse_click_string_to_left_right(s: &str) -> Result<ClickRange, String> {
    let mut split = s.split(',');
    if split.clone().count() != 2 {
        return Err("Mouse click spawn amount must be in format of '100,10'".to_string());
    }
    let left = split
        .next()
        .unwrap()
        .parse::<u32>()
        .map_err(|_| "Mouse click spawn amount must be a positive number".to_string())?;
    let right = split
        .next()
        .unwrap()
        .parse::<u32>()
        .map_err(|_| "Mouse click spawn amount must be a positive number".to_string())?;
    Ok(ClickRange(left, right))
}

#[derive(Clone, Copy, Debug, Default)]
struct Thing {
    velocity: (f32, f32),
    position: (f32, f32),
    texture_index: usize,
}

struct Simulator {
    things: Vec<Thing>,
    max_x: f32,
    max_y: f32,
}

impl Simulator {
    fn with_limit(max_x: f32, max_y: f32, capacity: usize) -> Self {
        Self {
            things: Vec::with_capacity(capacity),
            max_x,
            max_y,
        }
    }

    fn new(max_x: f32, max_y: f32) -> Self {
        Self::with_limit(max_x, max_y, 1000)
    }

    fn add_thing(&mut self, velocity: (f32, f32), position: (f32, f32), texture_index: usize) {
        self.things.push(Thing {
            velocity,
            position,
            texture_index,
        });
    }

    fn update(&mut self) {
        for thing in self.things.iter_mut() {
            thing.velocity.1 += 0.5; // gravity :3
            if (thing.position.0 + thing.velocity.0) > self.max_x {
                thing.velocity.0 *= -1.0;
                thing.position.0 = self.max_x;
            }
            if (thing.position.1 + thing.velocity.1) > self.max_y {
                thing.velocity.1 *= -0.8;
                thing.position.1 = self.max_y;
                if rand::random_bool(0.5) {
                    thing.velocity.1 -= 3.0 + rand::random_range(0.0..1.0) * 4.0;
                }
            }
            if (thing.position.0 + thing.velocity.0) < 0.0 {
                thing.velocity.0 *= -1.0;
                thing.position.0 = 0.0;
            }
            if (thing.position.1 + thing.velocity.1) < 0.0 {
                thing.velocity.1 *= -1.0;
                thing.position.1 = 0.0;
            } else {
                thing.position.0 += thing.velocity.0;
                thing.position.1 += thing.velocity.1;
            }
        }
    }
}

const DEFAULT_TEXTURE: &[u8] = include_bytes!("../wabbit_alpha.png");

const ROBOTO: &[u8] = include_bytes!("./assets/Roboto-Light.ttf");

#[allow(dead_code)]
/// soon™️
const TARGET_VIDEO_FRAMERATE: u32 = 60;

fn main() {
    better_panic::Settings::new()
        .lineno_suffix(true)
        .verbosity(better_panic::Verbosity::Full)
        .install();
    let cli = Cli::parse();
    if cli.list_gpu_renderers {
        println!("Available GPU renderers:");
        sdl3::render::drivers().enumerate().for_each(|(i, r)| {
            /* let mut flags = vec![];
            if r.flags & 0x00000001 != 0 {
                flags.push("Software Fallback");
            }
            if r.flags & 0x00000002 != 0 {
                flags.push("Hardware Accelerated");
            }
            if r.flags & 0x00000004 != 0 {
                flags.push("Present Vsync");
            }
            if r.flags & 0x00000008 != 0 {
                flags.push("Target Texture");
            }
            println!("{}: Renderer: {}", i + 1, r.name);
            println!("  Texture Formats Supported: {:?}", r.texture_formats);
            println!("  Max Texture Width: {}", r.max_texture_width);
            println!("  Max Texture Height: {}", r.max_texture_height);
            println!("  Rendering Capability: {}", flags.join(", ")); */
            println!("{}: Renderer: {}", i + 1, r);
            println!();
        });
        return;
    }

    let ctx = sdl3::init().unwrap();
    let video = ctx.video().unwrap();
    let font_ctx = sdl3::ttf::init().unwrap();

    /* let mut ffmpeg = None;
    if cli.record {
        ffmpeg = Some(ffmpeg::VideoRecorder::new("out.mp4", cli.width, cli.height, TARGET_VIDEO_FRAMERATE));
    } */

    let window = video
        .window(
            "Twodime - Stress Testing for GPU Renderers (And maybe Rust)",
            cli.width,
            cli.height,
        )
        .position_centered()
        .build()
        .unwrap();
    let mut canvas;
    if let Some(renderer) = cli.selected_gpu_renderer {
        let renderer_string =
            sdl3::render::drivers().collect::<Vec<String>>()[renderer as usize - 1].clone();
        canvas = sdl3::render::create_renderer(
            window,
            Some(CString::new(renderer_string).unwrap().as_c_str()),
        )
        .unwrap();
    } else {
        canvas = sdl3::render::create_renderer(window, None).unwrap();
    }
    let mut event_pump = ctx.event_pump().unwrap();

    let sdl3_supported_texture_formats = ["JPG", "PNG", "WEBP", "TIF"];

    let texture_creator = canvas.texture_creator();
    let mut textures = Vec::new();

    if let Some(texture_folder) = cli.texture_folder {
        for file in std::fs::read_dir(texture_folder).unwrap() {
            let file = file.unwrap();
            let path = file.path();
            if path.is_file() {
                let ext = path.extension().unwrap().to_str().unwrap();
                if sdl3_supported_texture_formats.contains(&ext) {
                    let texture: sdl3::render::Texture<'_> =
                        texture_creator.load_texture(path).unwrap();
                    let queried = texture.query();
                    if cli.texture_width != 0 {
                        let aspect_ratio = (queried.height as f32 * cli.texture_width as f32)
                            / queried.width as f32;
                        textures.push((texture, (cli.texture_width as f32, aspect_ratio)));
                    } else if cli.texture_height != 0 {
                        let aspect_ratio = (queried.width as f32 * cli.texture_height as f32)
                            / queried.height as f32;
                        textures.push((texture, (aspect_ratio, cli.texture_height as f32)));
                    } else {
                        textures.push((texture, (queried.width as f32, queried.height as f32)));
                    }
                }
            }
        }
    } else {
        let loaded = texture_creator.load_texture_bytes(DEFAULT_TEXTURE).unwrap();
        let queried = loaded.query();
        if cli.texture_width != 0 {
            // let aspect_ratio = (queried.width as f32 * cli.texture_height as f32) / queried.height as f32;
            let aspect_ratio =
                (queried.height as f32 * cli.texture_width as f32) / queried.width as f32;

            textures.push((loaded, (cli.texture_width as f32, aspect_ratio)));
        } else if cli.texture_height != 0 {
            let aspect_ratio =
                (queried.width as f32 * cli.texture_height as f32) / queried.height as f32;

            textures.push((loaded, (aspect_ratio, cli.texture_height as f32)));
        } else {
            textures.push((loaded, (queried.width as f32, queried.height as f32)));
        }
    }

    textures.windows(2).for_each(|h| {
        if h[0].1.0 != h[1].1.0 {
            panic!("Textures are not the same width");
        }
        if h[0].1.0 != h[1].1.0 {
            panic!("Textures are not the same height");
        }
    }); // is one texture just works with this?
    // oh wait
    // > The windows overlap. If the slice is shorter than size, the iterator returns no values.
    // lmao im dumb

    let queried = textures[0].1;

    assert!(
        !(queried.0 >= cli.width as f32 && queried.1 >= cli.height as f32),
        "Texture(s) is bigger than (or equal to) the window ({}x{})",
        queried.0,
        queried.1
    );

    // fps stuff
    let mut ft = std::time::Instant::now(); // frame time
    let mut fc = 0; // frame count
    let mut fps = 0.0; // frame per sec
    let mut mf = 0.0; // maximum fps
    let mut lf = 0.0; // minimum fps (shows on screen)
    let mut lpf = 0.0; // act as a cache
    let mut lft = std::time::Instant::now(); // minimum frame refresh time thingy

    println!("Loaded {} textures", textures.len());

    let max_x = cli.width as f32 - textures[0].1.0;
    let max_y = cli.height as f32 - textures[0].1.1;

    let simulator = Arc::new(parking_lot::RwLock::new(Simulator::new(max_x, max_y)));

    let fps_font = font_ctx
        .load_font_from_iostream(sdl3::iostream::IOStream::from_bytes(ROBOTO).unwrap(), 13.0)
        .unwrap();

    let exit_flag = Arc::new(AtomicBool::new(false));
    let (timer_sender, timer_receiver) = std::sync::mpsc::channel();
    spawn_thingy_spawner(
        simulator.clone(),
        exit_flag.clone(),
        cli.texture_spawn_time,
        textures.len(),
        cli.texture_limit,
        cli.spawn_by_click,
        timer_sender,
    );

    #[cfg(debug_assertions)]
    let mut current_mouse_x = 0.0;
    #[cfg(debug_assertions)]
    let mut current_mouse_y = 0.0;

    let mut last_physics_timer = std::time::Duration::new(0, 0);

    let mut vertexes = Vec::with_capacity((4 * cli.texture_limit) as usize);
    let mut indices = Vec::with_capacity((6 * cli.texture_limit) as usize);

    for i in 0..cli.texture_limit {
        let base = (i * 4) as u16;
                indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base + 1]);
    }

    'main_loop: loop {
        let frame_time = std::time::Instant::now();
        for event in event_pump.poll_iter() {
            match event {
                sdl3::event::Event::Quit { .. } => {
                    exit_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                    break 'main_loop;
                }
                #[cfg(debug_assertions)]
                sdl3::event::Event::MouseMotion { x, y, .. } => {
                    current_mouse_x = x as f32;
                    current_mouse_y = y as f32;
                }
                sdl3::event::Event::MouseButtonDown {
                    mouse_btn, x, y, ..
                } => {
                    let spawn_amount = if mouse_btn == sdl3::mouse::MouseButton::Left {
                        cli.mouse_click_spawn_amount.0
                    } else {
                        cli.mouse_click_spawn_amount.1
                    };
                    let pos = (
                        x as f32 - textures[0].1.0 / 2.0,
                        y as f32 - textures[0].1.1 / 2.0,
                    );
                    for _ in 0..spawn_amount {
                        let i = if textures.len() > 1 {
                            rand::random_range(0..textures.len() - 1)
                        } else {
                            0
                        };
                        {
                            simulator.write().add_thing(random_velocity(), pos, i);
                        }
                    }
                }
                _ => {}
            }
        }
        let mut render_time = std::time::Instant::now();
        canvas.clear();

        let mut geometry_time = None;
        if cli.draw_as_geometry {
            let geo_time = std::time::Instant::now();
            for (i, thing) in simulator.read().things.iter().enumerate() {
                let texture = &textures[thing.texture_index];
                let (texture_width, texture_height) = texture.1;
                let (texture_x, texture_y) = thing.position;
                let x = texture_x;
                let y = texture_y;
                vertexes.extend_from_slice(&[
                    Vertex {
                        color: sdl3::pixels::Color::WHITE.into(),
                        position: FPoint::new(x, y),
                        tex_coord: FPoint::new(0.0, 0.0),
                    },
                    Vertex {
                        color: sdl3::pixels::Color::WHITE.into(),
                        position: FPoint::new(x + texture_width, y),
                        tex_coord: FPoint::new(1.0, 0.0),
                    },
                    Vertex {
                        color: sdl3::pixels::Color::WHITE.into(),
                        position: FPoint::new(x, y + texture_height),
                        tex_coord: FPoint::new(0.0, 1.0),
                    },
                    Vertex {
                        color: sdl3::pixels::Color::WHITE.into(),
                        position: FPoint::new(x + texture_width, y + texture_height),
                        tex_coord: FPoint::new(1.0, 1.0),
                    },
                ]);
            }
            geometry_time = Some(geo_time.elapsed());
            render_time = std::time::Instant::now();
            canvas.render_geometry(&vertexes, Some(&textures[0].0), &indices[..(vertexes.len() / 4) * 6]).unwrap();
            
        } else {
            for thing in simulator.read().things.iter() {
                let texture = &textures[thing.texture_index];
                let (texture_width, texture_height) = texture.1;
                let (texture_x, texture_y) = thing.position;
                let x = texture_x;
                let y = texture_y;

                canvas
                    .copy(
                        &texture.0,
                        None,
                        sdl3::render::FRect::new(x, y, texture_width, texture_height),
                    )
                    .unwrap();
            }
        }
        let render_time = render_time.elapsed();

        if let Ok(t) = timer_receiver.try_recv() {
            last_physics_timer = t
        }

        let fps_text = fps_font
            .render(&format!("FPS: {}", truncate(fps, 2)))
            .shaded(sdl3::pixels::Color::WHITE, sdl3::pixels::Color::BLACK)
            .unwrap();
        let mf_text = fps_font
            .render(&format!("Maximum FPS: {}", truncate(mf, 2)))
            .shaded(sdl3::pixels::Color::WHITE, sdl3::pixels::Color::BLACK)
            .unwrap();
        let lfp_text = fps_font
            .render(&format!("Minimum FPS: {}", truncate(lf, 2)))
            .shaded(sdl3::pixels::Color::WHITE, sdl3::pixels::Color::BLACK)
            .unwrap();
        let added_render_latency_text = fps_font
            .render(&("Frame Time: ".to_string() + &number(frame_time.elapsed())))
            .shaded(sdl3::pixels::Color::WHITE, sdl3::pixels::Color::BLACK)
            .unwrap();
        let thingy_count = fps_font
            .render(&format!("Thingy Count: {}", simulator.read().things.len()))
            .shaded(sdl3::pixels::Color::WHITE, sdl3::pixels::Color::BLACK)
            .unwrap();
        let frame_time_breakdown = fps_font
            .render(&format!(
                "Frame Time Breakdown: Physics {} + Geometry {} + Rendering {}",
                number(last_physics_timer),
                if let Some(t) = geometry_time {
                    number(t)
                } else {
                    "N/A".to_string()
                },
                number(render_time)
            ))
            .shaded(sdl3::pixels::Color::WHITE, sdl3::pixels::Color::BLACK)
            .unwrap();
        canvas
            .copy(
                &texture_creator
                    .create_texture_from_surface(&fps_text)
                    .unwrap(),
                None,
                sdl3::render::FRect::new(
                    (canvas.output_size().unwrap().0 - fps_text.width()) as f32,
                    0.0,
                    fps_text.width() as f32,
                    fps_text.height() as f32,
                ),
            )
            .unwrap();
        canvas
            .copy(
                &texture_creator
                    .create_texture_from_surface(&mf_text)
                    .unwrap(),
                None,
                sdl3::rect::Rect::new(
                    (canvas.output_size().unwrap().0 - mf_text.width()) as i32,
                    fps_text.height() as i32 + 10,
                    mf_text.width(),
                    mf_text.height(),
                ),
            )
            .unwrap();
        canvas
            .copy(
                &texture_creator
                    .create_texture_from_surface(&lfp_text)
                    .unwrap(),
                None,
                sdl3::rect::Rect::new(
                    (canvas.output_size().unwrap().0 - lfp_text.width()) as i32,
                    mf_text.height() as i32 + fps_text.height() as i32 + 20,
                    lfp_text.width(),
                    lfp_text.height(),
                ),
            )
            .unwrap();

        canvas
            .copy(
                &texture_creator
                    .create_texture_from_surface(&added_render_latency_text)
                    .unwrap(),
                None,
                sdl3::rect::Rect::new(
                    (canvas.output_size().unwrap().0 - added_render_latency_text.width()) as i32,
                    (fps_text.height() + mf_text.height() + lfp_text.height() + 30) as i32,
                    added_render_latency_text.width(),
                    added_render_latency_text.height(),
                ),
            )
            .unwrap();
        canvas
            .copy(
                &texture_creator
                    .create_texture_from_surface(&thingy_count)
                    .unwrap(),
                None,
                sdl3::rect::Rect::new(
                    (canvas.output_size().unwrap().0 - thingy_count.width()) as i32,
                    (fps_text.height()
                        + mf_text.height()
                        + lfp_text.height()
                        + 40
                        + thingy_count.height()) as i32,
                    thingy_count.width(),
                    thingy_count.height(),
                ),
            )
            .unwrap();
        canvas
            .copy(
                &texture_creator
                    .create_texture_from_surface(&frame_time_breakdown)
                    .unwrap(),
                None,
                sdl3::rect::Rect::new(
                    (canvas.output_size().unwrap().0 - frame_time_breakdown.width()) as i32,
                    (fps_text.height()
                        + mf_text.height()
                        + lfp_text.height()
                        + 50
                        + thingy_count.height()
                        + frame_time_breakdown.height()) as i32,
                    frame_time_breakdown.width(),
                    frame_time_breakdown.height(),
                ),
            )
            .unwrap();
        #[cfg(debug_assertions)]
        {
            canvas.set_draw_color(sdl3::pixels::Color::CYAN);
            canvas
                .fill_rect(sdl3::rect::Rect::new(
                    current_mouse_x as i32,
                    current_mouse_y as i32,
                    100,
                    100,
                ))
                .unwrap();
            canvas.set_draw_color(sdl3::pixels::Color::BLACK);
        }
        canvas.present();
        fc += 1;
        let elapsed_time = ft.elapsed();
        if elapsed_time.as_secs() >= 1 {
            fps = fc as f64 / elapsed_time.as_secs_f64();
            fc = 0;
            ft = std::time::Instant::now();
            if fps > mf {
                mf = fps
            } else if fps < lpf {
                lpf = fps
            }
        }
        let elapsed_time = lft.elapsed();
        if elapsed_time.as_secs() >= 3 {
            lf = lpf;
            lpf = fps;
            lft = std::time::Instant::now();
        }
        vertexes.clear();
    }
}

fn spawn_thingy_spawner(
    simulator: Arc<parking_lot::RwLock<Simulator>>,
    exit_flag: Arc<AtomicBool>,
    spawn_time: f32,
    texture_counts: usize,
    texture_limit: u32,
    update_only: bool,
    timer_channel: std::sync::mpsc::Sender<std::time::Duration>,
) {
    std::thread::spawn(move || {
        let mut timer = std::time::Instant::now();
        let sim_time = 1.0 / 60.0;
        let mut sim_time_timer = std::time::Instant::now();
        while !exit_flag.load(std::sync::atomic::Ordering::Relaxed) {
            let mut locked = simulator.write();
            if timer.elapsed().as_secs_f32() >= spawn_time
                && locked.things.len() < texture_limit as usize
                && !update_only
            {
                timer = std::time::Instant::now();
                let pos = ((locked.max_x / 2.0), (locked.max_y / 2.0));
                let i = if texture_counts > 1 {
                    rand::random_range(0..texture_counts - 1)
                } else {
                    0
                };
                locked.add_thing(random_velocity(), pos, i);
            }
            if sim_time_timer.elapsed().as_secs_f32() >= sim_time {
                sim_time_timer = std::time::Instant::now();
                locked.update();
                timer_channel.send(sim_time_timer.elapsed()).unwrap();
            }
        }
    });
}

/// Truncate float with [`precision`] as how many digits you needed in final result
pub fn truncate(b: f64, precision: usize) -> f64 {
    f64::trunc(b * ((10 * precision) as f64)) / ((10 * precision) as f64)
}

pub fn random_velocity() -> (f32, f32) {
    (
        rand::random_range(-1.0..1.0) * 8.0,
        (rand::random_range(-1.0..1.0) * 5.0) - 2.5,
    )
}

fn number(b: std::time::Duration) -> String {
    let mut text = "".to_string();
    if b.as_secs_f64() >= 0.3 {
        text += &format!("{}s", b.as_secs());
    } else if b.as_millis() > 2 {
        text += &format!("{}ms", b.as_millis());
    } else if b.as_micros() != 0 {
        text += &format!("{}us", b.as_micros());
    } else {
        text += &format!("{}ns", b.as_nanos());
    }
    text
}
