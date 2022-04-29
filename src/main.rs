use std::ops::Range;
use std::time::Duration;
use std::collections::HashMap;
use std::fs;
use std::iter::StepBy;
use std::fs::File;
use serde::{Deserialize, Serialize};
use eframe::{epi, egui, epaint::Rgba};
use egui::Context;
use egui::plot::Legend;

use crate::epi::{Frame, Storage};

#[derive(Default)]
struct MyEguiApp {
    name : String,
    age : i32,
    frequency : f64,
    graph_data : HashMap<String, Vec<u64>>
}
struct Conv2DLayer{
    input_size : (u64, u64, u64),
    output_size : (u64, u64, u64),
    kernel : (u64, u64),
    stride : (u64, u64)
}

impl Conv2DLayer{
    fn new(input_size : (u64, u64, u64), kernel : (u64, u64), filters : u64, stride : (u64, u64)) -> Conv2DLayer{
        Conv2DLayer{
            input_size,
            output_size : (
                ((input_size.0 - 1)as f64 / stride.0 as f64).floor() as u64,
                ((input_size.1 - 1)as f64 / stride.1 as f64).floor() as u64,
                filters
            ),
            kernel,
            stride
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

struct VirtualMem{
    size : u64,
    word_size : u64
}

fn preload_memory(mem : &VirtualMem, num_rows : u64, row_size : u64) -> u64{
    // println!("Memory to be loaded {}", row_size * num_rows * mem.word_size);

    return (row_size * num_rows * mem.word_size / mem.size) * 50
}

fn example_plot(ui: &mut egui::Ui, frequency : f64, graph_data : &HashMap<String, Vec<u64>>) {
    use egui::plot::{Line, Value, Values};
    let n = 1024;
    let mut lines : Vec<Line> = Vec::new();
    for (k, v) in graph_data{
        lines.push(
            Line::new(
                Values::from_values_iter(
                    (0..v.len()).map(
                        |i| {
                            Value::new(i as f64, v[i] as f64)
                        }
                    )
                )
            ).name(k)
        );
    }
    egui::plot::Plot::new("example_plot")
        .data_aspect(1.0)
        .legend(
            Legend::default()
        )
        .include_x(0.0)
        .include_y(0.0)
        .show(ui, |
            plot_ui|
            {
                for line in lines {
                    plot_ui.line(line);
                }
            }
        );

}
fn example_inputs(window : &mut MyEguiApp, ui: &mut egui::Ui){

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
fn simulate_memory(){
    let total_mem : f64 = 2.5 * 2.0f64.powf(20.0);

    let activations_mem: VirtualMem = VirtualMem{
        size : (total_mem * 0.25) as u64,
        word_size : 4
    };
    let weights_mem: VirtualMem = VirtualMem{
        size : total_mem as u64 - activations_mem.size,
        word_size : 4
    };
    let input_x_range : Range<u64> = 17..18;
    let input_y_range : Range<u64> = 17..18;
    let input_channel_range : Range<u64>  = 1..1534;
    let kernel_x_range : Range<u64> = 1..2;
    let kernel_y_range : Range<u64> = 1..2;
    let filter_range : StepBy<Range<u64>> = (1..1025).step_by(128);
    let mut preloads_over_input_channels : Vec<u64> = Vec::new();
    let mut preloads_for_configs : HashMap<String, Vec<u64>> = HashMap::new();
    for input_x in input_x_range.clone(){
        for input_y in input_y_range.clone(){
            for filter in filter_range.clone(){
                for kernel_x in kernel_x_range.clone(){
                    for kernel_y in kernel_y_range.clone(){
                        for input_c in input_channel_range.clone(){
                            let activation_preloads : u64 = preload_memory(&activations_mem, input_y, input_x * input_c);
                            let weights_preloads : u64 = preload_memory(&weights_mem, filter, kernel_x * kernel_y * input_c);
                            preloads_over_input_channels.push(weights_preloads);
                        }
                        preloads_for_configs.insert(format!("In({}, {}), k({}, {}), fi {}", input_x, input_y, kernel_x, kernel_y, filter), preloads_over_input_channels.clone());
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
