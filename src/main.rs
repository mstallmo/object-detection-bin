use image::{GenericImageView, Rgba};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use ndarray::prelude::*;
use std::fs::File;
use std::io::Read;
use tensorflow::{Graph, ImportGraphDefOptions, Session, SessionOptions, SessionRunArgs, Tensor};

const MODEL: &str = "/home/mason/Workspace/object-detection-rs/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb";
const IMAGE_1: &str = "/home/mason/Workspace/object-detection-rs/test_images/image1.jpg";
const IMAGE_2: &str = "/home/mason/Workspace/object-detection-rs/test_images/image2.jpg";

fn main() {
    //initialize graph and session
    let mut graph = Graph::new();
    let mut proto = Vec::new();
    File::open(MODEL).unwrap().read_to_end(&mut proto).unwrap();
    graph
        .import_graph_def(&proto, &ImportGraphDefOptions::new())
        .unwrap();
    let mut sess = Session::new(&SessionOptions::new(), &graph).unwrap();

    //get handle to input tensor
    let image_tensor = graph.operation_by_name_required("image_tensor").unwrap();
    let mut image_1 = image::open(IMAGE_1).unwrap();
    let (width, height) = image_1.dimensions();

    //Read in image
    let pixel_array = image_1.raw_pixels();
    //TODO: convert height and width to usize
    let image_array = Array::from_shape_vec((636, 1024, 3), pixel_array).unwrap();
    let image_array_expanded = image_array.insert_axis(Axis(0));

    //create new tensor with the image data
    let mut input_image_tensor = Tensor::new(&[1, height as u64, width as u64, 3])
        .with_values(image_array_expanded.as_slice().unwrap())
        .unwrap();
    let mut step = SessionRunArgs::new();
    step.add_feed(&image_tensor, 0, &input_image_tensor);

    //output tensors: detection_classes, num_detections, detection_scores, detection_boxes
    let num_detections = graph.operation_by_name_required("num_detections").unwrap();
    let num_detections_token = step.request_output(&num_detections, 0);

    let classes = graph
        .operation_by_name_required("detection_classes")
        .unwrap();
    let classes_token = step.request_output(&classes, 0);

    let boxes = graph.operation_by_name_required("detection_boxes").unwrap();
    let boxes_token = step.request_output(&boxes, 0);

    let scores = graph
        .operation_by_name_required("detection_scores")
        .unwrap();
    let scores_token = step.request_output(&scores, 0);

    sess.run(&mut step).unwrap();

    let num_detections_tensor = step.fetch::<f32>(num_detections_token).unwrap();
    println!("{:?}", num_detections_tensor[0]);

    let classes_tensor = step.fetch::<f32>(classes_token).unwrap();
    println!("{:?}", classes_tensor.iter().collect::<Vec<_>>());

    let boxes_tensor = step.fetch::<f32>(boxes_token).unwrap();
    let dims = Dim(boxes_tensor
        .dims()
        .iter()
        .map(|&e| e as usize)
        .collect::<Vec<_>>());
    let boxes_array = Array::from_shape_vec((dims), boxes_tensor.iter().collect::<Vec<_>>())
        .unwrap()
        .remove_axis(Axis(0));
    for (index, element) in boxes_array.axis_iter(Axis(0)).enumerate() {
        println!("Index: {}", index);
        println!("Element: {:?}", element);
    }
    let first_box = boxes_array.axis_iter(Axis(0)).nth(0).unwrap();
    let box_slice = first_box.as_slice().unwrap();
    let min_y = *box_slice[0];
    let min_x = *box_slice[1];
    let max_y = *box_slice[2];
    let max_x = *box_slice[3];

    let white = Rgba([255u8, 255u8, 255u8, 255u8]);
    let bounding_box = Rect::at(
        (min_x * width as f32) as i32,
        (min_y * height as f32) as i32,
    )
    .of_size(
        ((max_x - min_x) * width as f32) as u32,
        ((max_y - min_y) * height as f32) as u32,
    );
    draw_hollow_rect_mut(&mut image_1, bounding_box, white);
    image_1
        .save("/home/mason/Workspace/object-detection-rs/test_output.jpg")
        .unwrap();

    let scores_tensor = step.fetch::<f32>(scores_token).unwrap();
    println!("{:?}", scores_tensor.iter().collect::<Vec<_>>());
}
