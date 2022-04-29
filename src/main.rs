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

#[derive(Serialize, Deserialize, Debug, Clone)]
struct CharacterisationParameters{
    min_iterations : u64,
    max_iterations : u64,
    error_margin : f64,
    confidence : f64
}
#[derive(Serialize, Deserialize, Debug, Clone)]
struct Conv2DParameters{
    filter : Vec<u64>,
    in_c: Vec<u64>,
    in_s : Vec<u64>,
    kx : Vec<u64>,
    ky : Vec<u64>,
    stride : Vec<u64>,
    maximum_complexity : u64
}
#[derive(Serialize, Deserialize, Debug, Clone)]
struct TimingConfig{
    characterisation_parameters : CharacterisationParameters,
    parameters : Conv2DParameters,
    modes : Vec<String>
}
#[derive(Serialize, Deserialize, Debug, Clone)]
struct FullConfig{
    hardware : HashMap<String, HashMap<String, HashMap<String, TimingConfig>>>
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
            // example_inputs(self, ui);
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


fn calc_activation_mem(layer : &Conv2DLayer) -> u64{
    // let input_activations : u64 = layer.input_size.0 * layer.input_size.1 * layer.input_size.2;
    // let output_activations : u64 = layer.output_size.0 * layer.output_size.1 * layer.output_size.2;
    let input_activations : u64 = layer.input_size.0 * layer.kernel.1 * layer.input_size.2;
    let output_activations : u64 = layer.output_size.0 * layer.kernel.1 * layer.output_size.2;
    return input_activations + output_activations;
}
fn calc_weights_mem(layer : &Conv2DLayer) -> u64{

    return layer.input_size.2 * layer.output_size.2 * layer.kernel.0 * layer.kernel.1 + layer.output_size.2;
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
    for (k, v) in &preloads_for_configs{
        println!("{}", k);
        println!("{:?}", v);
    }



    let mut app = MyEguiApp::default();
    app.graph_data = preloads_for_configs.clone();
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(Box::new(app), native_options);
}
fn main() {
    simulate_memory();
    let reader: File = File::open("./config.json").unwrap();
    let config : FullConfig = serde_json::from_reader(reader).unwrap();
    let conv2d_config : Conv2DParameters = config.hardware.get("MyriadX").unwrap().get("Conv2D").unwrap().get("timing").unwrap().parameters.clone();
    // println!("{:?}", conv2d_config);
    let mut num_examples_per_mem : HashMap<u64, u64> = HashMap::new();
    let mut max_mem = 0;
    for input_s in &conv2d_config.in_s{
        for input_c in &conv2d_config.in_c{
            for filter in &conv2d_config.filter{
                for kernel_x in &conv2d_config.kx{
                    for kernel_y in &conv2d_config.ky {
                        for stride in &conv2d_config.stride{

                            // let input_size : (u64, u64, u64) =
                            let conv_layer : Conv2DLayer = Conv2DLayer::new((*input_s, *input_s, *input_c), (*kernel_x, *kernel_y), *filter, (*stride, *stride));

                            let activation_mem : u64 = calc_activation_mem(&conv_layer);
                            let weights_mem : u64 = calc_weights_mem(&conv_layer);

                            let mem : u64 = activation_mem + weights_mem;
                            if mem > max_mem{
                                max_mem = mem;
                            }
                            match num_examples_per_mem.get(&mem){
                                None => {
                                    num_examples_per_mem.insert(mem, 1);
                                }
                                Some(previous) => {
                                    num_examples_per_mem.insert(mem, previous + 1);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    println!("{}", max_mem);
    let mut num_configs_per_example_count : HashMap<u64, u64> = HashMap::new();
    println!("{:?}", num_examples_per_mem.len());
    for (mem, count) in &num_examples_per_mem{
        match num_configs_per_example_count.get(&count){
            None => {
                num_configs_per_example_count.insert(*count, 1);
            }
            Some(previous) => {
                num_configs_per_example_count.insert(*count, previous + 1);
            }
        }
    }
    println!("{:?}", num_configs_per_example_count.get(&1));
    println!("{:?}", num_configs_per_example_count.get(&2));
    println!("{:?}", num_configs_per_example_count.get(&3));
}
