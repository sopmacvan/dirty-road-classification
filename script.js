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

function createNewModelHead() {
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

    let imagePaths = metadata.map(item => `${IMG_DIR}${item.filename}`);
    let labels = metadata.map(item => parseInt(item.label)); // Assuming labels are integers (0 or 1)
    tf.util.shuffleCombo(imagePaths, labels);

    // Load and preprocess images
    const imageFeatures = await Promise.all(imagePaths.map(async (filename) => {
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

async function makePredictions(imagePaths) {
    let preprocessedDataArray = await Promise.all(imagePaths.map(async (filename) => {
        return await loadAndPreprocessImage(filename);
    }));
    preprocessedDataArray = tf.stack(preprocessedDataArray, 0).squeeze(1)
    let predictions = await combinedModel.predict(preprocessedDataArray);
    predictions = predictions.dataSync()
    console.log("imagePaths", imagePaths);
    console.log("raw score", predictions);
    predictions = predictions.map(value => value > 0.5 ? 1 : 0);
    console.log("predictions", predictions);

    return predictions;
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
const MARKERS = document.querySelectorAll('.marker');
const ROADS_COUNT = MARKERS.length;


async function createRandomRoadObjects(count = 10, ) {
    const metadata = await fetch('./dataset/metadata.csv')
        .then(response => response.text())
        .then(data => Papa.parse(data, {header: true}))
        .then(result => result.data);

    let imagePaths = metadata.map(item => `${IMG_DIR}${item.filename}`);
    const roadNames = [
        'Maple Avenue', 'Oak Street', 'Cedar Lane', 'Main Street', 'Elm Road', 'Birch Avenue', 'Pine Street', 'Willow Lane', 'Hickory Road', 'Cherry Street',
        // 'Sycamore Lane', 'Ash Street', 'Cypress Road', 'Beech Avenue', 'Fir Street', 'Mulberry Lane', 'Poplar Road', 'Chestnut Street', 'Magnolia Lane', 'Holly Avenue',
        // 'Juniper Street', 'Cottonwood Lane', 'Basswood Road', 'Spruce Street', 'Locust Avenue', 'Catalpa Lane', 'Sassafras Road', 'Dogwood Street', 'Redwood Lane', 'Alder Avenue',
        // 'Pecan Street', 'Bamboo Road', 'Balsa Lane', 'Buckeye Street', 'Butternut Avenue', 'Cactus Lane', 'Camphor Street', 'Cashew Road', 'Cedar Avenue', 'Chinquapin Lane',
        // 'Cinnamon Street', 'Citron Road', 'Clover Avenue', 'Cocoa Lane', 'Coffee Street', 'Cola Road', 'Cork Avenue', 'Corncob Lane', 'Cotton Street', 'Cranberry Road',
        // 'Currant Avenue', 'Cypress Lane', 'Daisy Street', 'Dandelion Road', 'Date Avenue', 'Dragonfly Lane', 'Driftwood Street', 'Duckweed Road', 'Dusty Lane', 'Eagle Street',
        // 'Earthen Avenue', 'Ebony Road', 'Echo Lane', 'Eden Street', 'Elderberry Avenue', 'Elephant Road', 'Ember Lane', 'Emerald Street', 'Evergreen Avenue', 'Falcon Road',
        // 'Fern Lane', 'Fiddlehead Street', 'Fir Avenue', 'Flame Road', 'Flamingo Lane', 'Flint Street', 'Flower Avenue', 'Foggy Lane', 'Forest Street', 'Foxglove Road',
        // 'Frosted Avenue', 'Frosty Lane', 'Fruit Street', 'Fuchsia Avenue', 'Galaxy Lane', 'Garnet Street', 'Gazelle Road', 'Geranium Avenue', 'Ghost Lane', 'Ginger Street',
        // 'Glacier Avenue', 'Glade Lane', 'Glowing Street', 'Gobbler Road', 'Golden Avenue', 'Goose Lane', 'Gossamer Street', 'Granite Avenue', 'Gravity Lane', 'Greenbriar Street',
        // 'Greystone Avenue', 'Grotto Lane', 'Guava Street', 'Gulch Avenue', 'Gull Lane', 'Gumdrop Street', 'Gypsum Avenue', 'Harmony Lane', 'Haven Street', 'Hawthorn Avenue',
        // 'Haystack Lane', 'Heather Street', 'Heirloom Avenue', 'Hemlock Lane', 'Hickory Street', 'Highland Avenue', 'Horizon Lane', 'Huckleberry Street', 'Hummingbird Avenue', 'Hyacinth Lane',
        // 'Icicle Street', 'Idyllic Avenue', 'Indigo Lane', 'Infinite Street', 'Inkwell Avenue', 'Iron Lane', 'Island Street', 'Ivory Avenue', 'Jade Lane', 'Jazz Street',
        // 'Jubilee Avenue', 'Junction Lane', 'Juniper Street', 'Kaleidoscope Avenue', 'Kangaroo Lane', 'Keynote Street', 'Kingfisher Avenue', 'Kite Lane', 'Kitten Street', 'Kiwi Avenue',
        // 'Labyrinth Lane', 'Lagoon Street', 'Larkspur Avenue', 'Lattice Lane', 'Lavender Street', 'Legacy Avenue', 'Lighthouse Lane', 'Lilac Street', 'Lily Avenue', 'Lime Lane',
        // 'Linen Street', 'Lion Avenue', 'Lithium Lane', 'Lively Street', 'Log Lane', 'Lollipop Avenue', 'Lullaby Lane', 'Lunar Street', 'Luxury Avenue', 'Lyric Lane',
        // 'Magenta Street', 'Majestic Avenue', 'Mango Lane', 'Meadow Street', 'Melody Avenue', 'Midnight Lane', 'Mint Street', 'Misty Avenue', 'Monarch Lane', 'Moonbeam Street',
        // 'Morning Avenue', 'Mossy Lane', 'Moth Street', 'Mountain Avenue', 'Muse Lane', 'Mystic Street', 'Nectar Avenue', 'Nestling Lane', 'Noble Street', 'Nomad Avenue',
        // 'Nostalgia Lane', 'Nova Street', 'Nutmeg Avenue', 'Oasis Lane', 'Oat Street', 'Ocean Avenue', 'Opulent Lane', 'Oracle Street', 'Oregano Avenue', 'Origami Lane',
        // 'Ornate Street', 'Outpost Lane', 'Ovation Avenue', 'Oyster Lane', 'Paddle Street', 'Painter Avenue', 'Palace Lane', 'Panda Street', 'Panorama Avenue', 'Paprika Lane',
        // 'Parade Street', 'Parchment Avenue', 'Parlor Lane', 'Patriot Street', 'Peachy Avenue', 'Pebble Lane', 'Pegasus Street', 'Pendulum Avenue', 'Perfume Lane', 'Periscope Street',
        // 'Periwinkle Avenue', 'Petunia Lane', 'Piano Street', 'Piccolo Avenue', 'Pilgrim Lane', 'Pillow Street', 'Pioneer Avenue', 'Pixel Lane', 'Placid Street', 'Platinum Avenue',
        // 'Playful Lane', 'Pleasant Street', 'Polar Avenue', 'Polished Lane', 'Pollen Street', 'Polymer Avenue', 'Pond Lane', 'Porcelain Street', 'Portal Avenue', 'Portico Lane',
        // 'Posh Street', 'Prairie Avenue', 'Prism Lane', 'Prosperity Street', 'Puddle Avenue', 'Pulse Lane', 'Pumpkin Street', 'Purity Avenue', 'Puzzle Lane', 'Quasar Street',
        // 'Queen Avenue', 'Quilted Lane', 'Quintessence Street', 'Quiver Avenue', 'Radiant Lane', 'Rainbow Street', 'Ramble Avenue', 'Ranch Lane', 'Raven Street', 'Rebel Avenue',
        // 'Reef Lane', 'Reflection Street', 'Regal Avenue', 'Relic Lane', 'Renegade Street', 'Resolute Avenue', 'Retro Lane', 'Reverie Street', 'Rhythm Avenue', 'Ripple Lane',
        // 'Rivulet Street', 'Roam Avenue', 'Roar Lane', 'Rooster Street', 'Rosy Avenue', 'Royal Lane', 'Ruby Street', 'Rustic Avenue', 'Safari Lane', 'Sage Street',
        // 'Sailor Avenue', 'Sapphire Lane', 'Savvy Street', 'Scenic Avenue', 'Sculpture Lane', 'Seaside Street', 'Serenade Avenue', 'Shady Lane', 'Shimmer Street', 'Silent Avenue',
        // 'Silhouette Lane', 'Silver Street', 'Simple Avenue', 'Sizzle Lane', 'Skylark Street', 'Sleek Avenue', 'Slice Lane', 'Smile Street', 'Snowflake Avenue', 'Solar Lane',
        // 'Soothing Street', 'Soulful Avenue', 'Spark Lane', 'Spectacle Street', 'Spiral Avenue', 'Splendid Lane', 'Spooky Street', 'Spring Avenue', 'Sprout Lane', 'Starry Street',
        // 'Steady Avenue', 'Stellar Lane', 'Stone Street', 'Stunning Avenue', 'Sublime Lane', 'Summer Street', 'Summit Avenue', 'Sunny Lane', 'Sunset Street', 'Superb Avenue',
        // 'Surge Lane', 'Surreal Street', 'Sweet Avenue', 'Swift Lane', 'Symphony Street', 'Tapestry Avenue', 'Tea Lane', 'Tranquil Street', 'Traverse Avenue', 'Treasure Lane',
        // 'Twilight Street', 'Umbra Avenue', 'Uplift Lane', 'Vibrant Street', 'Vista Avenue', 'Vivid Lane', 'Voyage Street', 'Wander Avenue', 'Whimsical Lane', 'Whisper Street',
        // 'Willow Avenue', 'Wishful Lane', 'Wonder Street', 'Woven Avenue', 'Yonder Lane', 'Zenith Street', 'Zeppelin Avenue', 'Zest Lane'
    ];

    const randomIndices = Array.from({length: count}, () => Math.floor(Math.random() * (imagePaths.length - 1)));
    imagePaths = randomIndices.map(index => imagePaths[index])

    let roadObjects = [];
    for (let i=0; i<count; i++){
        let roadObject = new Road(roadNames[i], imagePaths[i])
        roadObjects.push(roadObject);
    }
    return roadObjects;

}

async function selectRandomRoadImagesAndClassify(){
    // Create road objects and get image paths
    let roads = await createRandomRoadObjects(ROADS_COUNT);
    let allImages = roads.map(road => road.imagePath);
    let allImagesFlattened = [].concat(...allImages);

    // Make predictions
    let predictions = await makePredictions(allImagesFlattened)

    return [roads, predictions];
}
async function handleRefreshClick() {
    showNotification('Refreshing images...');

    selectRandomRoadImagesAndClassify()
        .then(([roads, predictions]) => {
            updateMarkers(MARKERS, roads, predictions);
            hideNotification();
        })
        .catch(error => {
            console.error("An error occurred:", error);
            hideNotification();
        });
}

function showNotification(message) {
    const notification = document.getElementById('notification');
    notification.textContent = message;
    notification.classList.remove('hidden');
    setTimeout(() => hideNotification(), 3000); // Hide after 3 seconds
}

function hideNotification() {
    const notification = document.getElementById('notification');
    notification.classList.add('hidden');
}

// Add click event listener to each marker
async function updateMarkers(markers, roads, predictions) {
    markers.forEach((marker, index) => {
        let road = roads[index];
        let isDirty = predictions[index];

        // Set background image based on prediction value
        marker.style.backgroundImage = `url(${isDirty ? TRASH_RED_ICON : TRASH_GREEN_ICON})`;

        marker.addEventListener('click', () => {
            // Update sidebar content
            document.getElementById('road-name').innerText = road.roadName;
            document.getElementById('road-image').src = road.imagePath;
            document.getElementById('cleanliness-status').innerText = isDirty ? "DIRTY" : "CLEAN";
            // Show sidebar
            document.getElementById('sidebar').style.display = 'block';
        });
    });
}
// Close sidebar when clicking outside of it
document.addEventListener('click', (event) => {
    if (!event.target.closest('#sidebar') && !event.target.classList.contains('marker')) {
        document.getElementById('sidebar').style.display = 'none';
    }
});
// Add click event to refresh button
document.addEventListener('DOMContentLoaded', function () {
    const refreshButton = document.getElementById('refresh-button');
    refreshButton.addEventListener('click', handleRefreshClick);
});


loadModel()
    .then(() => selectRandomRoadImagesAndClassify())
    .then(([roads, predictions]) => {
        updateMarkers(MARKERS, roads, predictions);
    })


//
//
//
// Function to shuffle images and update markers
// async function shuffleImagesAndPredict() {
//     // Shuffle the images
//     const shuffledImages = tf.util.shuffle(ALL_IMAGES_FLATTENED);
//
//     // Select the first 10 shuffled images
//     const selectedImages = shuffledImages.slice(0, 10);
//
//     // Make predictions for the selected images
//     const predictions = await makePredictions(selectedImages);
//
//     // Update markers with new predictions and images
//     updateMarkers(markers, predictions);
// }

// Set interval to shuffle images every 5 seconds
// setInterval(async () => {
//     await shuffleImagesAndPredict();
// }, 5000);
