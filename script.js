// ------------------------------
// All Models Related
// ------------------------------
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
let mobilenet;
let mobileNetBase;
let model;
let combinedModel;

async function loadMobileNetFeatureModel() {
    const URL = 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/mobilenet-v2/model.json';
    mobilenet = await tf.loadLayersModel(URL);

    const layer = mobilenet.getLayer('global_average_pooling2d_1');
    mobileNetBase = tf.model({inputs: mobilenet.inputs, outputs: layer.output});

    // Warm up the model by passing zeros through it once.
    tf.tidy(function () {
        mobileNetBase.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
        console.log("Base model loaded successfully!")
    });
}

function createNewModelHead(){
    model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [1280], units: 128, activation: 'relu'}));
    model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));

    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    console.log("New model head loaded successfully!")
}

function preprocessImage(img) {
    const tensor = tf.tidy(() => {
        // Convert the image to a tensor
        const imageTensor = tf.browser.fromPixels(img);
        // Resize the image to match the input dimensions of MobileNet
        const resizedImg = tf.image.resizeBilinear(imageTensor, [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH]);
        // Normalize the image
        const normalizedImg = resizedImg.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
        // Expand dimensions to match the expected shape [1,224,224,3]
        return normalizedImg.expandDims(0);
    });

    return tensor;
}

// Function to load and preprocess an image
async function loadAndPreprocessImage(filename) {
    const img = new Image();
    img.src = filename;
    await img.decode(); // Ensure the image is loaded
    return preprocessImage(img); // has shape [1,224,224,3]
}

async function prepareDataset() {
    const metadata = await fetch('./dataset/metadata.csv')
        .then(response => response.text())
        .then(data => Papa.parse(data, {header: true}))
        .then(result => result.data);

    let filenames = metadata.map(item => `${IMG_DIR}${item.filename}`);
    let labels = metadata.map(item => parseInt(item.label)); // Assuming labels are integers (0 or 1)
    tf.util.shuffleCombo(filenames, labels);

    // Load and preprocess images
    const imageFeatures = await Promise.all(filenames.map(async (filename) => {
        // Load and preprocess the image
        const preprocessedImage = await loadAndPreprocessImage(filename);
        // Make predictions using mobileNetBase and apply squeeze
        return mobileNetBase.predict(preprocessedImage).squeeze();
    }));
    console.log("Data preparation complete!")

    return [imageFeatures, labels];
}

function logProgress(epoch, logs) {
    console.log('Data for epoch ' + epoch, logs);
}

async function trainModel() {
    const epochs = 10;
    const batchSize = 5;
    const [imageFeatures, labels] = await prepareDataset();

    // Convert the array of tensors to a single tensor
    const inputsAsTensor = tf.stack(imageFeatures);
    const outputsAsTensor = tf.tensor1d(labels, 'int32');
    // Train the model
    await model.fit(inputsAsTensor, outputsAsTensor, {
        batchSize,
        epochs,
        callbacks: {onEpochEnd: logProgress},
    });
    inputsAsTensor.dispose()
    outputsAsTensor.dispose()

    console.log('Training complete!');
}

async function saveModel() {
    combinedModel = tf.sequential();
    combinedModel.add(mobileNetBase);
    combinedModel.add(model);

    await combinedModel.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy'
    });
    await combinedModel.save('downloads://my-model')

    console.log('Model saved!')
}

async function loadModel() {
    // Load the model architecture and weights
    combinedModel = await tf.loadLayersModel('./model/my-model.json');
    combinedModel.summary()

    console.log('Model loaded!');
}

async function makePrediction(filenames) {
    let preprocessedDataArray = await Promise.all(filenames.map(async (filename) => {
        return await loadAndPreprocessImage(filename);
    }));
    preprocessedDataArray = tf.stack(preprocessedDataArray,0).squeeze(1)
    let prediction = await combinedModel.predict(preprocessedDataArray);
    prediction = prediction.dataSync()
    console.log("filenames", filenames);
    console.log("raw score", prediction);
    prediction = prediction.map(value => value > 0.5 ? 1 : 0);
    console.log("prediction", prediction);

    return prediction;
}

//// Start training
// loadMobileNetFeatureModel()
//     .then(() => createNewModelHead())
//     .then(() => trainModel())
//     .then(() => saveModel())


// ------------------------------
// HTML Events
// ------------------------------
const IMG_DIR = "./dataset/Images/"
const TRASH_RED_ICON = "./assets/icons/trash_red.png"
const TRASH_GREEN_ICON = "./assets/icons/trash_green.png"
const ROADS = [
    {name: "Clean Road", image: IMG_DIR + "clean_1.jpg",},
    {name: "Clean Road", image: IMG_DIR + "clean_2.jpg",},
    {name: "Clean Road", image: IMG_DIR + "clean_3.jpg",},
    {name: "Clean Road", image: IMG_DIR + "clean_4.jpg",},
    {name: "Clean Road", image: IMG_DIR + "clean_5.jpg",},
    {name: "Clean Road", image: IMG_DIR + "clean_6.jpg",},

    {name: "Dirty Road", image: IMG_DIR + "dirty_1.jpg",},
    {name: "Dirty Road", image: IMG_DIR + "dirty_2.jpg",},
    {name: "Dirty Road", image: IMG_DIR + "dirty_3.jpg",},
    {name: "Dirty Road", image: IMG_DIR + "dirty_4.jpg",},
    {name: "Dirty Road", image: IMG_DIR + "dirty_5.jpg",},
    {name: "Dirty Road", image: IMG_DIR + "dirty_6.jpg",}
    // Add more ROADS as needed
];
const ALL_IMAGES = ROADS.map(road => road.image)
const ALL_IMAGES_FLATTENED = [].concat(ALL_IMAGES)

async function updateMarkers(markers, dirtinessClassifications){
    markers.forEach((marker, index) => {
        let road = ROADS[index];
        let prediction = dirtinessClassifications[index];

        // Set background image based on prediction value
        marker.style.backgroundImage = `url(${prediction === 1 ? TRASH_RED_ICON : TRASH_GREEN_ICON})`;

        marker.addEventListener('click', () => {
            // Update sidebar content
            document.getElementById('road-name').innerText = road.name;
            document.getElementById('road-image').src = road.image;
            // Show sidebar
            document.getElementById('sidebar').style.display = 'block';
        });
    });
}

// Add click event listener to each marker
const markers = document.querySelectorAll('.marker');
loadModel()
    .then(() => makePrediction(ALL_IMAGES_FLATTENED))
    .then(predictions => updateMarkers(markers, predictions));

// Close sidebar when clicking outside of it
document.addEventListener('click', (event) => {
    if (!event.target.closest('#sidebar') && !event.target.classList.contains('marker')) {
        document.getElementById('sidebar').style.display = 'none';
    }
});

//
//
//
async function shuffleImagesAndPredict() {
    // Shuffle the images
    const shuffledImages = tf.util.shuffle(ALL_IMAGES_FLATTENED);

    // Make predictions for the shuffled images
    const predictions = await makePrediction(shuffledImages);

    // Update markers with new predictions and images
    updateMarkers(markers, predictions);
}

// Set interval to shuffle images every 5 seconds
setInterval(async () => {
    await shuffleImagesAndPredict();
}, 5000); 
