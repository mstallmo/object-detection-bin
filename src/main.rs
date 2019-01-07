use image::GenericImageView;
use ndarray::prelude::*;
use ndarray::ShapeBuilder;
use std::fs::File;
use std::io::Read;
use tensorflow::{Graph, ImportGraphDefOptions, Session, SessionOptions, Tensor};

const MODEL: &str = "/Users/user/Workspace/object-detection-rs/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb";
const IMAGE_1: &str = "/Users/user/Workspace/object-detection-rs/test_images/image1.jpg";
const IMAGE_2: &str = "/Users/user/Workspace/object-detection-rs/test_images/image2.jpg";

fn main() {
    let mut graph = Graph::new();
    let mut proto = Vec::new();
    File::open(MODEL).unwrap().read_to_end(&mut proto).unwrap();
    graph
        .import_graph_def(&proto, &ImportGraphDefOptions::new())
        .unwrap();
    let sess = Session::new(&SessionOptions::new(), &graph).unwrap();
    let image_tensor = graph.operation_by_name_required("image_tensor").unwrap();
    let image_1 = image::open(IMAGE_1).unwrap();
    let (width, height) = image_1.dimensions();
    let input_image_tensor = <Tensor<u8>>::new(&[1, height as u64, width as u64, 3]);
    println!("{:?}", input_image_tensor);
    //    let a = Array::from_shape_vec((1, 3), image_1.raw_pixels()).unwrap();
    //    println!("{:?}", a);
    let a = Array::from_shape_vec((2, 3).strides((3, 1)), vec![1, 2, 3, 4, 5, 6]).unwrap();
    println!("{:?}", a);
    println!("{}", image_1.raw_pixels().iter().count() / 3);
    let pixel_array = image_1.raw_pixels();
    println!("{:?}", pixel_array);
    let test = Array::from_vec(pixel_array);
    println!("{:?}", test);
//    let image_array =
//        Array::from_shape_vec((636, 1024).strides((3, 1)), pixel_array).unwrap();
//    println!("{:?}", image_array);
}
