// builtin
use std::error::Error;

// external

// internal
use crate::types::HandwrittenExample;

pub fn load_data_from_csv(
    path: &str,
    offset: usize,
    rows: usize,
) -> Result<Vec<HandwrittenExample>, Box<dyn Error>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)?;

    let mut data: Vec<HandwrittenExample> = Vec::new();
    for result in reader.deserialize().skip(offset).take(rows) {
        let record: Vec<u16> = result?;
        let labelled: HandwrittenExample = HandwrittenExample::new(record)?;
        data.push(labelled);
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use model::trainer::examples::SupervisedExample;

    use crate::import_csv::load_data_from_csv;

    #[test]
    fn load_data_test() {
        let data = load_data_from_csv("../data/mnist_train.csv", 0, 2).expect("Data load failed!");

        println!("Data: {:?}", data);

        println!("Label: {:?}", data[0].get_response());
    }
}
