// builtin
use std::error::Error;

// external

// internal
use crate::types::LabelledData;

pub fn load_data_from_csv(path: &str) -> Result<Vec<LabelledData>, Box<dyn Error>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)?;

    let mut data: Vec<LabelledData> = Vec::new();
    for result in reader.deserialize().take(5) {
        let record: Vec<u16> = result?;
        let labelled: LabelledData = LabelledData::new(record)?;
        data.push(labelled);
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use crate::import_csv::load_data_from_csv;

    #[test]
    fn load_data_test() {
        let data = load_data_from_csv("../data/mnist_train.csv").expect("Data load failed!");

        println!("{:?}", data);
    }
}
