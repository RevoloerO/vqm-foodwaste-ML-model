const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const csv = require('csv-parser');

// Load data from CSV file
async function loadCSVData(filePath) {
    return new Promise((resolve, reject) => {
        const data = [];
        fs.createReadStream(filePath)
            .pipe(csv())
            .on('data', (row) => {
                data.push({
                    year: parseFloat(row.year),
                    foodWaste: parseFloat(row.foodWaste) // Adjust column names as needed
                });
            })
            .on('end', () => resolve(data))
            .on('error', (error) => reject(error));
    });
}

// Prepare data for ML model
function prepareData(data) {
    const inputs = data.map(d => d.year);
    const outputs = data.map(d => d.foodWaste);

    return {
        xs: tf.tensor2d(inputs, [inputs.length, 1]),
        ys: tf.tensor2d(outputs, [outputs.length, 1])
    };
}

// Train the model
async function trainModel(xs, ys) {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
    
    await model.fit(xs, ys, { epochs: 500 });
    return model;
}

// Predict future food waste
async function predictFuture(model, year) {
    const input = tf.tensor2d([year], [1, 1]);
    const prediction = model.predict(input);
    return prediction.dataSync()[0];
}

// Main function
async function main() {
    const filePath = 'food_waste_data.csv'; // Ensure this file exists with correct data
    const rawData = await loadCSVData(filePath);
    const { xs, ys } = prepareData(rawData);
    
    console.log('Training model...');
    const model = await trainModel(xs, ys);
    console.log('Model trained!');
    
    const futureYear = 2030;
    const predictedWaste = await predictFuture(model, futureYear);
    console.log(`Predicted food waste for ${futureYear}: ${predictedWaste}`);
}

main().catch(console.error);
