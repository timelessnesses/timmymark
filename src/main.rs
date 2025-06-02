use std::sync::{atomic::AtomicBool, Arc, LazyLock, RwLock};
use rand::Rng;

use clap::Parser;
use rand::SeedableRng;
use sdl2::{self, image::LoadTexture};
#[derive(clap::Parser)]
#[command(author = "timelessnesses", about = "SDL2 (Rust) Bunnymark")]
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

    /// Texture limit number (default: 10000)
    #[arg(long, default_value_t = 10000)]
    pub texture_limit: u32,

    /// Custom 2D texture folder (default: None (program's included textures))
    #[arg(long)]
    pub texture_folder: Option<std::path::PathBuf>,

    /// Texture width, the aspect ratio will be respected, if this value was set, texture_height will be ignored (default: 128)
    #[arg(long, default_value_t = 64, conflicts_with = "texture_height")]
    pub texture_width: u32,

    /// Texture height, the aspect ratio will be respected if this value was set, texture_width will be ignored (default: 128)
    #[arg(long, default_value_t = 0, conflicts_with = "texture_width")]
    pub texture_height: u32,

    /// Texture time to spawn (default: 0.01 seconds)
    #[arg(long, default_value_t = 0.01)]
    pub texture_spawn_time: f32,
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
    fn new(max_x: f32, max_y: f32) -> Self {
        Self {
            things: Vec::new(),
            max_x,
            max_y,
        }
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

const DEFAULT_TEXTURE: &[u8] = include_bytes!("../dumb_idiot.png");

const ROBOTO: &[u8] = include_bytes!("./assets/Roboto-Light.ttf");

fn main() {
    better_panic::Settings::new()
        .lineno_suffix(true)
        .verbosity(better_panic::Verbosity::Full)
        .install();
    let cli = Cli::parse();
    if cli.list_gpu_renderers {
        println!("Available GPU renderers:");
        if let Some((i, r)) = sdl2::render::drivers().enumerate().next() {
            let mut flags = vec![];
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
            println!("  Rendering Capability: {}", flags.join(", "));
            println!();
            return;
        }
    }

    let ctx = sdl2::init().unwrap();
    let video = ctx.video().unwrap();
    let _image = sdl2::image::init(sdl2::image::InitFlag::all()).unwrap();
    let font_ctx = sdl2::ttf::init().unwrap();

    let window = video
        .window(
            "Twodime - Stress Testing for GPU Renderers (And maybe Rust)",
            cli.width,
            cli.height,
        )
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().accelerated();
    if let Some(renderer) = cli.selected_gpu_renderer {
        canvas = canvas.index(renderer - 1);
    }

    let mut canvas = canvas.build().unwrap();
    let mut event_pump = ctx.event_pump().unwrap();

    let sdl2_supported_texture_formats = ["JPG", "PNG", "WEBP", "TIF"];

    let texture_creator = canvas.texture_creator();
    let mut textures = Vec::new();

    if let Some(texture_folder) = cli.texture_folder {
        for file in std::fs::read_dir(texture_folder).unwrap() {
            let file = file.unwrap();
            let path = file.path();
            if path.is_file() {
                let ext = path.extension().unwrap().to_str().unwrap();
                if sdl2_supported_texture_formats.contains(&ext) {
                    let texture: sdl2::render::Texture<'_> =
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

    let simulator = Arc::new(RwLock::new(Simulator::new(max_x, max_y)));

    let fps_font = font_ctx
        .load_font_from_rwops(sdl2::rwops::RWops::from_bytes(ROBOTO).unwrap(), 13)
        .unwrap();

    let exit_flag = Arc::new(AtomicBool::new(false));

    spawn_thingy_spawner(simulator.clone(), exit_flag.clone(), cli.texture_spawn_time, textures.len(), cli.texture_limit);

    let mut current_mouse_x = 0.0;
    let mut current_mouse_y = 0.0;

    'main_loop: loop {
        let frame_time = std::time::Instant::now();
        for event in event_pump.poll_iter() {
            match event {
                sdl2::event::Event::Quit { .. } => {
                    exit_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                    break 'main_loop;
                },
                sdl2::event::Event::MouseMotion { x, y, .. } => {
                    current_mouse_x = x as f32;
                    current_mouse_y = y as f32;
                }
                _ => { }
            }
        }
        canvas.clear();
        for thing in simulator.read().unwrap().things.iter() {
            let texture = &textures[thing.texture_index];
            let (texture_width, texture_height) = texture.1;
            let (texture_x, texture_y) = thing.position;
            let x = texture_x;
            let y = texture_y;
            canvas
                .copy(
                    &texture.0,
                    None,
                    Some(sdl2::rect::Rect::new(
                        x as i32,
                        y as i32,
                        texture_width as u32,
                        texture_height as u32,
                    )),
                )
                .unwrap();
        }

        let fps_text = fps_font
            .render(&format!("FPS: {}", truncate(fps, 2)))
            .shaded(sdl2::pixels::Color::WHITE, sdl2::pixels::Color::BLACK)
            .unwrap();
        let mf_text = fps_font
            .render(&format!("Maximum FPS: {}", truncate(mf, 2)))
            .shaded(sdl2::pixels::Color::WHITE, sdl2::pixels::Color::BLACK)
            .unwrap();
        let lfp_text = fps_font
            .render(&format!("Minimum FPS: {}", truncate(lf, 2)))
            .shaded(sdl2::pixels::Color::WHITE, sdl2::pixels::Color::BLACK)
            .unwrap();
        let added_render_latency_text = fps_font
            .render(&format!(
                "Render Latency: {}ns (nanoseconds!)",
                frame_time.elapsed().as_nanos()
            ))
            .shaded(sdl2::pixels::Color::WHITE, sdl2::pixels::Color::BLACK)
            .unwrap();
        let thingy_count = fps_font
            .render(&format!("Thingy Count: {}", simulator.read().unwrap().things.len()))
            .shaded(sdl2::pixels::Color::WHITE, sdl2::pixels::Color::BLACK)
            .unwrap();
        canvas
            .copy(
                &texture_creator
                    .create_texture_from_surface(&fps_text)
                    .unwrap(),
                None,
                sdl2::rect::Rect::new(
                    (canvas.output_size().unwrap().0 - fps_text.width()) as i32,
                    0,
                    fps_text.width(),
                    fps_text.height(),
                ),
            )
            .unwrap();
        canvas
            .copy(
                &texture_creator
                    .create_texture_from_surface(&mf_text)
                    .unwrap(),
                None,
                sdl2::rect::Rect::new(
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
                sdl2::rect::Rect::new(
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
                sdl2::rect::Rect::new(
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
                sdl2::rect::Rect::new(
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
        #[cfg(debug_assertions)]
        {
            canvas.set_draw_color(sdl2::pixels::Color::CYAN);
            canvas.fill_rect(sdl2::rect::Rect::new(current_mouse_x as i32, current_mouse_y as i32, 100, 100)).unwrap();
            canvas.set_draw_color(sdl2::pixels::Color::BLACK);
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
    }
}

fn spawn_thingy_spawner(simulator: Arc<RwLock<Simulator>>, exit_flag: Arc<AtomicBool>, spawn_time: f32, texture_counts: usize, texture_limit: u32) {
    std::thread::spawn(move || {
        let mut timer = std::time::Instant::now();
        let sim_time = 1.0/120.0;
        let mut sim_time_timer = std::time::Instant::now();
        while !exit_flag.load(std::sync::atomic::Ordering::Relaxed) {
            let mut locked = simulator.write().unwrap();
            if timer.elapsed().as_secs_f32() >= spawn_time && locked.things.len() < texture_limit as usize {
                timer = std::time::Instant::now();
                let pos = ((locked.max_x / 2.0) as f32, (locked.max_y / 2.0) as f32);
                let i = if texture_counts > 1 {
                        rand::random_range(0..texture_counts - 1)
                    } else {
                        0
                    };
                locked.add_thing(
                    (
                        rand::random_range(0.0..1.0) * 8.0,
                        (rand::random_range(0.0..1.0) * 5.0) - 2.5,
                    ),
                    pos,
                    i,
                );
            }
            if sim_time_timer.elapsed().as_secs_f32() >= sim_time {
                sim_time_timer = std::time::Instant::now();
                locked.update();
            }
        }
    });
}

/// Truncate float with [`precision`] as how many digits you needed in final result
pub fn truncate(b: f64, precision: usize) -> f64 {
    f64::trunc(b * ((10 * precision) as f64)) / ((10 * precision) as f64)
}
