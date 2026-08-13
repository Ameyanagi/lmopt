use anyhow::{Context, Result};
use lmopt::least_squares;

fn main() -> Result<()> {
    let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let ys = [3.1, 5.2, 6.9, 9.1, 11.0, 13.1];

    let fit = least_squares(&[1.0, 1.0], |parameters| {
        let [slope, intercept] = parameters else { return Vec::new() };
        xs.iter()
            .zip(ys)
            .map(|(x, y)| slope * x + intercept - y)
            .collect()
    })
    .context("failed to fit the line")?;

    let parameters = fit.parameters();
    println!("Fitted line: y = {:.4}*x + {:.4}", parameters[0], parameters[1]);
    Ok(())
}
