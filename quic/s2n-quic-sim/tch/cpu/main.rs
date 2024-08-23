// This example illustrates how to use a PyTorch model trained and exported using the
// Python JIT API.
// See https://pytorch.org/tutorials/advanced/cpp_export.html for more details.
use anyhow::{bail, Result};
// use tch::vision::imagenet;
use tch::{kind, Tensor, Kind};

pub fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let (model_file) = match args.as_slice() {
        [_, m] => (m.to_owned()),
        _ => bail!("usage: main model.pt"),
    };
    // Load the image file and resize it to the usual imagenet dimension of 224x224.
    //let image = imagenet::load_image_and_resize224(image_file)?;

    // Load the Python saved module.
    let model = tch::CModule::load(model_file)?;

    // Apply the forward pass of the model to get the logits.
    let input = Tensor::randn([1, 64, 8], kind::FLOAT_CPU);
    //let output = image.unsqueeze(0).apply(&model).softmax(-1, tch::Kind::Float);
    let output = input.apply(&model);
    println!("output: {output}");

    // Print the top 5 categories for this image.
    //for (probability, class) in imagenet::top(&output, 5).iter() {
    //    println!("{:50} {:5.2}%", class, 100.0 * probability)
    //}
    Ok(())
}
