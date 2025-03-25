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
            console.log(item); // Log feature values 
            
            // Handle missing or invalid data
            if (!item.year || !item.foodWaste) {
                throw new Error('Invalid data');
            }
            
            return {
                xs: [item.foodWaste], // Replace with actual feature names
                ys: item.year // Replace with actual label name
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
    model.add(tf.layers.dense({ units: 32, activation: 'relu', inputShape: [1] })); // Adjust inputShape based on your features
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'linear' })); // Adjust output units and activation based on your problem

    model.compile({
        optimizer: tf.train.adam(),
        loss: 'meanSquaredError', // Adjust loss function based on your problem
        metrics: ['mae']
    });

    return model;
}

// Train the model
async function trainModel(model, data) {
    const xs = tf.tensor2d(data.map(item => item.xs));
    const ys = tf.tensor2d(data.map(item => item.ys), [data.length, 1]);

    await model.fit(xs, ys, {
        epochs: 50,
        batchSize: 32,
        validationSplit: 0.2
    });
}

// Evaluate the model
async function evaluateModel(model, data) {
    const xs = tf.tensor2d(data.map(item => item.xs));
    const ys = tf.tensor2d(data.map(item => item.ys), [data.length, 1]);

    const result = model.evaluate(xs, ys);
    const loss = result[0].dataSync();
    const mae = result[1].dataSync();

    console.log(`Evaluation result - Loss: ${loss}, MAE: ${mae}`);
}

// Make predictions with the model
async function makePredictions(model, newData) {
    const xs = tf.tensor2d(newData.map(item => item.xs));
    const predictions = model.predict(xs);
    const predictedValues = predictions.dataSync();
    newData.forEach((item, index) => {
        console.log(`Input: ${item.xs}, Predicted Year: ${predictedValues[index]}`);
    });

    return predictedValues;
}

// Main function
async function run() {
    try {
        const data = await loadData();
        console.log('Data loaded and preprocessed:', data);
        
        // Split data into training and testing sets
        const trainSize = Math.floor(data.length * 0.8);
        const trainData = data.slice(0, trainSize);
        const testData = data.slice(trainSize);

        const model = createModel();
        await trainModel(model, trainData);

        console.log('Model training complete');

        // Evaluate the model
        await evaluateModel(model, testData);

        // Make predictions on new data
        const newData = [
            { xs: [2013] }, // Replace with actual new data
            { xs: [2020] }  // Replace with actual new data
        ];
        await makePredictions(model, newData);

    } catch (error) {
        console.error('Error in run function:', error);
    }
}

run();