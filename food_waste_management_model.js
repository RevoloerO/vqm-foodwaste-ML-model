const tf = require('@tensorflow/tfjs');
const csvUrl = 'file://C:/Users/quyen/OneDrive/Documents/GitHub/vqm-foodwaste-ML-model/food_waste_management.csv';

// Load and preprocess data
async function loadData() {
    try {
        const data = await tf.data.csv(csvUrl, {
            hasHeader: true,
        }).toArray();
        console.log('Data loaded successfully:', data);
        
        // Preprocess data
        const processedData = data.map(item => {
            return {
                xs: [
                    item.year
                ],
                ys: [
                    item.generation,
                    item.composted,
                    item.other_food_management,
                    item.combustion_with_energy_recovery,
                    item.landfilled
                ] // Predict all features
            };
        });
        return processedData;
    } catch (error) {
        console.error('Error loading or processing data:', error);
        throw error;
    }
}

// Define the model
function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 32, activation: 'relu', inputShape: [1] })); // 1 feature (year)
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 5, activation: 'linear' })); // 5 outputs (one for each feature)

    model.compile({
        optimizer: tf.train.adam(),
        loss: 'meanSquaredError',
        metrics: ['mae']
    });

    return model;
}

// Train the model
async function trainModel(model, data) {
    const xs = tf.tensor2d(data.map(item => item.xs));
    const ys = tf.tensor2d(data.map(item => item.ys), [data.length, 5]);

    await model.fit(xs, ys, {
        epochs: 50,
        batchSize: 32,
        validationSplit: 0.2
    });
}

// Evaluate the model
async function evaluateModel(model, data) {
    const xs = tf.tensor2d(data.map(item => item.xs));
    const ys = tf.tensor2d(data.map(item => item.ys), [data.length, 5]);

    const result = model.evaluate(xs, ys);
    const loss = result[0].dataSync();
    const mae = result[1].dataSync();

    console.log(`Evaluation result - Loss: ${loss}, MAE: ${mae}`);
}

// Evaluate the model and calculate percentage accuracy
function evaluateModelAccuracy(predictedValues, actualValues) {
    console.log('Feature | Percentage Accuracy');
    console.log('--------|--------------------');
    const numFeatures = predictedValues[0].length;

    for (let i = 0; i < numFeatures; i++) {
        let totalError = 0;
        let totalActual = 0;

        predictedValues.forEach((predicted, index) => {
            totalError += Math.abs(predicted[i] - actualValues[index][i]);
            totalActual += actualValues[index][i];
        });

        const accuracy = ((1 - totalError / totalActual) * 100).toFixed(2);
        const featureName = ['Generation', 'Composted', 'Other Food Management', 'Combustion with Energy Recovery', 'Landfilled'][i];
        console.log(`${featureName} | ${accuracy}%`);
    }
}

// Make predictions with the model
async function makePredictions(model, newData) {
    const xs = tf.tensor2d(newData.map(item => item.xs), [newData.length, 1]); // Ensure input shape matches [null, 1]
    const predictions = model.predict(xs);
    const predictedValues = predictions.arraySync();
    newData.forEach((item, index) => {
        console.log(`Input Year: ${item.xs}, Predicted Features: ${predictedValues[index]}`);
    });

    return predictedValues;
}

// Print predicted values as a table
function printPredictedValues(predictedValues, years) {
    console.log('Year | Generation | Composted | Other Food Management | Combustion with Energy Recovery | Landfilled');
    console.log('-----|------------|-----------|------------------------|----------------------------------|-----------');
    predictedValues.forEach((values, index) => {
        console.log(
            `${years[index]} | ${values[0].toFixed(2)} | ${values[1].toFixed(2)} | ${values[2].toFixed(2)} | ${values[3].toFixed(2)} | ${values[4].toFixed(2)}`
        );
    });
}

// Print table showing changes (cumulative sum of predicted values and previous years' values)
function printChangesTable(predictedValues, years, previousYearValues) {
    console.log('Year | Generation | Composted | Other Food Management | Combustion with Energy Recovery | Landfilled');
    console.log('-----|------------|-----------|------------------------|----------------------------------|-----------');
    let cumulativeValues = [...previousYearValues]; // Start with the previous year's values

    predictedValues.forEach((values, index) => {
        cumulativeValues = cumulativeValues.map((val, i) => val + values[i]); // Add predicted values to cumulative values
        console.log(
            `${years[index]} | ${cumulativeValues[0].toFixed(2)} | ${cumulativeValues[1].toFixed(2)} | ${cumulativeValues[2].toFixed(2)} | ${cumulativeValues[3].toFixed(2)} | ${cumulativeValues[4].toFixed(2)}`
        );
    });
}

// Main function
async function run() {
    try {
        const data = await loadData();
        console.log('Data loaded and preprocessed:', data);
        
        // Get the latest year's values from the dataset
        const latestYearValues = data[data.length - 1].ys;

        // Split data into training and testing sets
        const trainSize = Math.floor(data.length * 0.8);
        const trainData = data.slice(0, trainSize);
        const testData = data.slice(trainSize);

        const model = createModel();
        await trainModel(model, trainData);

        console.log('Model training complete');

        // Evaluate the model
        await evaluateModel(model, testData);

        // Make predictions on test data for evaluation
        const testXs = testData.map(item => item.xs);
        const testYs = testData.map(item => item.ys);
        const testPredictions = await makePredictions(model, testXs.map(xs => ({ xs })));

        // Evaluate percentage accuracy
        evaluateModelAccuracy(testPredictions, testYs);

        // Make predictions on new data
        const newData = [
            { xs: [2020] },
            { xs: [2021] },
            { xs: [2022] },
            { xs: [2023] },
            { xs: [2024] }, // Example input year for prediction
            { xs: [2025] }  // Example input year for prediction
        ];
        const predictions = await makePredictions(model, newData);

        // Print predictions as a table
        const years = newData.map(item => item.xs[0]);
        printPredictedValues(predictions, years);

        // Print changes table
        printChangesTable(predictions, years, latestYearValues);

    } catch (error) {
        console.error('Error in run function:', error);
    }
}

run();