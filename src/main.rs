use crate::epi::{Frame, Storage};
use eframe::{egui, epaint::Rgba, epi};
use egui::plot::Legend;
use egui::Context;
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::iter::StepBy;
use std::ops::Range;
use std::time::Duration;

#[derive(Default)]
struct MyEguiApp {
    name: String,
    age: i32,
    frequency: f64,
    graph_data: HashMap<String, Vec<u64>>,
}
struct Conv2DLayer {
    input_size: (u64, u64, u64),
    output_size: (u64, u64, u64),
    kernel: (u64, u64),
    stride: (u64, u64),
}

impl Conv2DLayer {
    fn new(
        input_size: (u64, u64, u64),
        kernel: (u64, u64),
        filters: u64,
        stride: (u64, u64),
    ) -> Conv2DLayer {
        Conv2DLayer {
            input_size,
            output_size: (
                ((input_size.0 - 1) as f64 / stride.0 as f64).floor() as u64,
                ((input_size.1 - 1) as f64 / stride.1 as f64).floor() as u64,
                filters,
            ),
            kernel,
            stride,
        }
    }
}

impl epi::App for MyEguiApp {
    fn update(&mut self, ctx: &egui::Context, frame: &epi::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Hello World!");
            example_inputs(self, ui);
            example_plot(ui, self.frequency, &self.graph_data);
        });
    }
    fn setup(&mut self, _ctx: &Context, _frame: &Frame, _storage: Option<&dyn Storage>) {
        self.frequency = 1.0;
    }
    fn name(&self) -> &str {
        "My egui App"
    }
}

struct VirtualMem {
    size: u64,
    word_size: u64,
}

fn preload_memory(mem: &VirtualMem, num_rows: u64, row_size: u64) -> u64 {
    // println!("Memory to be loaded {}", row_size * num_rows * mem.word_size);

    return (row_size * num_rows * mem.word_size / mem.size) * 50;
}

fn pdf(x: f64, y: f64) -> f64 {
    const SDX: f64 = 0.1;
    const SDY: f64 = 0.1;
    const A: f64 = 5.0;
    let x = x as f64 / 10.0;
    let y = y as f64 / 10.0;
    A * (-x * x / 2.0 / SDX / SDX - y * y / 2.0 / SDY / SDY).exp()
}
fn example_plot(
    ui: &mut egui::Ui,
    frequency: f64,
    graph_data: &HashMap<String, Vec<u64>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::gif("./example.gif", (600, 400), 100)?.into_drawing_area();

    for pitch in 0..157 {
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("2D Gaussian PDF", ("sans-serif", 20))
            .build_cartesian_3d(
                (-3.0..3.0) as Range<f64>,
                (0.0..6.0) as Range<f64>,
                (-3.0..3.0) as Range<f64>,
            )?;
        chart.with_projection(|mut p| {
            p.pitch = 1.57 - (1.57 - pitch as f64 / 50.0).abs();
            p.scale = 0.7;
            p.into_matrix() // build the projection matrix
        });

        chart
            .configure_axes()
            .light_grid_style(BLACK.mix(0.15))
            .max_light_lines(3)
            .draw()?;

        chart.draw_series(
            SurfaceSeries::xoz(
                (-15..=15).map(|x| x as f64 / 5.0),
                (-15..=15).map(|x| x as f64 / 5.0),
                pdf,
            )
            .style_func(&|&v| {
                (&HSLColor(240.0 / 360.0 - 240.0 / 360.0 * v / 5.0, 1.0, 0.7)).into()
            }),
        )?;

        root.present()?;
    }

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");

    Ok(())
}
fn example_inputs(window: &mut MyEguiApp, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.label("Your name: ");
        ui.add(egui::TextEdit::singleline(&mut window.name));
    });
    ui.add(egui::Slider::new(&mut window.frequency, 0.0f64..=120.0f64).text("age"));
    if ui.button("Click each year").clicked() {
        window.frequency += 1.0;
    }
    ui.label(format!("Hello '{}', age {}", window.name, window.age));
}
fn simulate_memory() {
    let total_mem: f64 = 2.5 * 2.0f64.powf(20.0);

    let activations_mem: VirtualMem = VirtualMem {
        size: (total_mem * 0.25) as u64,
        word_size: 4,
    };
    let weights_mem: VirtualMem = VirtualMem {
        size: total_mem as u64 - activations_mem.size,
        word_size: 4,
    };
    let input_x_range: Range<u64> = 17..18;
    let input_y_range: Range<u64> = 17..18;
    let input_channel_range: Range<u64> = 1..1534;
    let kernel_x_range: Range<u64> = 1..2;
    let kernel_y_range: Range<u64> = 1..2;
    let filter_range: StepBy<Range<u64>> = (1..1025).step_by(128);
    let mut preloads_over_input_channels: Vec<u64> = Vec::new();
    let mut preloads_for_configs: HashMap<String, Vec<u64>> = HashMap::new();
    for input_x in input_x_range.clone() {
        for input_y in input_y_range.clone() {
            for filter in filter_range.clone() {
                for kernel_x in kernel_x_range.clone() {
                    for kernel_y in kernel_y_range.clone() {
                        for input_c in input_channel_range.clone() {
                            let activation_preloads: u64 =
                                preload_memory(&activations_mem, input_y, input_x * input_c);
                            let weights_preloads: u64 =
                                preload_memory(&weights_mem, filter, kernel_x * kernel_y * input_c);
                            preloads_over_input_channels.push(weights_preloads);
                        }
                        preloads_for_configs.insert(
                            format!(
                                "In({}, {}), k({}, {}), fi {}",
                                input_x, input_y, kernel_x, kernel_y, filter
                            ),
                            preloads_over_input_channels.clone(),
                        );
                        preloads_over_input_channels = Vec::new();
                    }
                }
            }
        }
    }
    let mut app = MyEguiApp::default();
    app.graph_data = preloads_for_configs.clone();
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(Box::new(app), native_options);
}
fn main() {
    simulate_memory();
}
