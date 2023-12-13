const IMG_DIR = "./assets/dataset/Images/"
const TRASH_RED_ICON = "./assets/icons/trash_red.png"
const TRASH_GREEN_ICON = "./assets/icons/trash_green.png"
const ROADS = [
    {name: "Clean Road", image: IMG_DIR + "clean_1.jpg",},
    {name: "Clean Road", image: IMG_DIR + "clean_2.jpg",},
    {name: "Clean Road", image: IMG_DIR + "clean_3.jpg",},
    {name: "Clean Road", image: IMG_DIR + "clean_4.jpg",},
    {name: "Clean Road", image: IMG_DIR + "clean_5.jpg",},
    {name: "Dirty Road", image: IMG_DIR + "dirty_1.jpg",},
    {name: "Dirty Road", image: IMG_DIR + "dirty_2.jpg",},
    {name: "Dirty Road", image: IMG_DIR + "dirty_3.jpg",},
    {name: "Dirty Road", image: IMG_DIR + "dirty_4.jpg",},
    {name: "Dirty Road", image: IMG_DIR + "dirty_5.jpg",},
    // Add more ROADS as needed
];

// Add click event listener to each marker
let markers = document.querySelectorAll('.marker');
markers.forEach((marker, index) => {
    let road = ROADS[index];
    let prediction = 0

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

// Close sidebar when clicking outside of it
document.addEventListener('click', (event) => {
    if (!event.target.closest('#sidebar') && !event.target.classList.contains('marker')) {
        document.getElementById('sidebar').style.display = 'none';
    }
});


// ALL MODEL RELATED
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
        let answer = mobileNetBase.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
        console.log("Base model loaded successfully!")
    });
}

model = tf.sequential();
model.add(tf.layers.dense({inputShape: [1280], units: 128, activation: 'relu'}));
model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));

// Compile the model with the defined optimizer and specify a loss function to use.
model.compile({
    optimizer: 'adam',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
});
console.log("New model head loaded successfully!")
model.summary()

// Function to preprocess the image and get the tensor
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
    let result;
    img.src = filename;
    await img.decode(); // Ensure the image is loaded
    result = preprocessImage(img);
    result = mobileNetBase.predict(result).squeeze();
    return result
}
async function prepareDataset(){
    const metadata = await fetch('./assets/dataset/metadata.csv')
        .then(response => response.text())
        .then(data => Papa.parse(data, {header: true}))
        .then(result => result.data);

    let filenames = metadata.map(item => `./assets/dataset/Images/${item.filename}`);
    let labels = metadata.map(item => parseInt(item.label)); // Assuming labels are integers (0 or 1)
    tf.util.shuffleCombo(filenames, labels);

    // Load and preprocess images
    const imageFeatures = await Promise.all(filenames.map(loadAndPreprocessImage));
    console.log("Data preparation complete!")

    return [imageFeatures, labels];
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

    combinedModel.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy'
    });
    combinedModel.save('downloads://my-model')

    console.log('Model saved!')
}

// Start training

loadMobileNetFeatureModel()
    .then(() => trainModel())
    .then(() => saveModel());


function logProgress(epoch, logs) {
    console.log('Data for epoch ' + epoch, logs);
}