const express = require('express');
const tf = require("@tensorflow/tfjs-node-gpu"); // Ensure using GPU version
const faceapi = require("@vladmandic/face-api/dist/face-api.node-gpu.js");
const path = require('path');
const { createCanvas, loadImage } = require('canvas');

const fs_promise = require('fs').promises;
const cors = require('cors');
const winston = require('winston');
const sharp = require('sharp');
const mysql = require('mysql');

const db = mysql.createConnection({
  host: 'localhost',
  user: 'root',
  password: '',
  database: 'moi'
});

db.connect((err) => {
  if (err) {
    console.error('Error connecting to MySQL database:', err);
    return;
  }
  console.log('Connected to MySQL database');
});


const modelPath = './model'; // Path to the model directory
const minConfidence = 0.6; // Minimum confidence for face detection
const maxResults = 5; // Maximum number of results to return
const distanceThreshold = 0.5; // Distance threshold for face recognition
let optionsSSDMobileNet; // Options for SSD MobileNet

let labelInfo = {};
let faceMatcher;

// .... Log file Configs ....//
const logger = winston.createLogger({
  level: 'error', // Set the logging level to 'error' or 'info' based on your requirements
  format: winston.format.simple(),
  transports: [
    new winston.transports.File({ filename: 'face_api_logs.log' }) // Log errors to a file named 'error.log'
  ]
});


// .... Check and set the backend ....//
async function checkAndSetBackend() {
  console.log(`Current backend before setting: ${tf.getBackend()}`);

  await tf.setBackend('tensorflow'); // 'tensorflow' for using bindings to the Python TensorFlow library
  await tf.ready();

  console.log(`Current backend after setting: ${tf.getBackend()}`);
}

// .... Load label info from file ....//
async function loadLabelInfo() {
  const filePath = path.join(__dirname, 'label_info/labelInfo.json');
  try {
    const data = await fs_promise.readFile(filePath, 'utf8');
    const labelArray = JSON.parse(data);
    labelInfo = labelArray.reduce((acc, item) => {
      acc[item.id] = { name: item.name, status: item.status, position: item.position };
      return acc;
    }, {});
    console.log('Label info loaded and transformed:', labelInfo);
  } catch (error) {
    console.error('Failed to load label info:', error);
    labelInfo = {};
  }
}

// .... Initialize FaceAPI and load models ....// 
async function initializeFaceAPI() { // gpu
  await Promise.all([
    faceapi.tf.setBackend('tensorflow'),
    faceapi.tf.ready(),
    faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath),
    faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath),
    faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath),
  ]);
  optionsSSDMobileNet = new faceapi.SsdMobilenetv1Options({ minConfidence, maxResults });
}

// Load labeled images asynchronously
async function loadLabeledImages(labels) {
  if (!Array.isArray(labels) || labels.length === 0) {
    throw new Error("Labels must be a non-empty array.");
  }

  // Load descriptors for each label
  const promises = labels.map(async label => {
    const descriptions = [];
    for (let i = 0; i <= 5; i++) {
      console.log(label, i);
      descriptions.push(await loadFaceDescriptor(label, i));
    }
    return new faceapi.LabeledFaceDescriptors(label, descriptions);
  });

  return Promise.all(promises);
}

async function loadFaceDescriptor(label, index) { // gpu
  try {
    const imgPath = `./labeled_images/${label}/${index}.jpeg`;

    console.log('Image Path:', imgPath);

    const tensorImage = await loadImageAndConvertToTensor(imgPath);

    if (!tensorImage) {
      throw new Error(`Failed to load image or detect face for ${label}/${index}`);
    }

    const detection = await faceapi.detectSingleFace(tensorImage).withFaceLandmarks().withFaceDescriptor();

    // Dispose of the tensor image after it's no longer needed
    tensorImage.dispose();
    if (!detection) {
      throw new Error(`Face not detected for ${label}/${index}`);
    }
    return detection.descriptor;
  } catch (error) {
    console.error('Error in loadFaceDescriptor:', error);
    logger.error(`Error during server setup: ${error.message}`);

    throw error; // Rethrow the error for higher-level handling
  }
}

async function loadImageAndConvertToTensor(imgPath) {
  try {
    const image = await loadImage(imgPath);
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, image.width, image.height);

    const tensor = tf.tidy(() => {
      const buffer = canvas.toBuffer();
      const decode = faceapi.tf.node.decodeImage(buffer, 3);
      return faceapi.tf.expandDims(decode, 0);
    });

    return tensor;
  } catch (error) {
    console.error('Error during image loading and conversion for path', imgPath, ':', error);
    logger.error(`Error during image loading and conversion for path ${imgPath}: ${error.message}`);
    throw error;
  }
}

// .... Setup the server and handle requests .... //
async function setupServer() {
  try {
    const app = express(); // Create an Express app 
    const port = 3005;

    // Enable CORS for all requests
    app.use(cors()); //
    app.use(express.json({ limit: '10mb' }));
    app.use(express.urlencoded({ limit: '10mb', extended: true }));

    app.post('/new_face', cors(), handle_new_face);
    app.post('/face_recognition', cors(), handle_face_recognition);
    app.use((req, res, next) => {
      req.on('aborted', () => {
        console.error('Request aborted by the client:', req.url);
      });
      next();
    });

    app.listen(port, () => {
      console.log(`Server running on http://localhost:${port}`);
    }); // Start the server on the specified port

  } catch (error) {
    console.error('Error during server setup:', error);
    logger.error(`Error during server setup: ${error.message}`);
  }
}

async function handle_new_face(req, res) {
  try {
    const { name, images, status, position } = req.body;

    if (!name || !images || !status || !position || images.length === 0) {
      return res.status(400).send('Invalid input: name, status, position and images are required.');
    }

    // Create a unique ID using the current timestamp
    const id = Date.now().toString();

    // Create a directory for the user if it doesn't already exist
    const dirPath = path.join(__dirname, 'labeled_images', id);
    await fs_promise.mkdir(dirPath, { recursive: true }).catch(err => {
      if (err.code !== 'EEXIST') throw err; // Only throw if error is not because directory already exists
    });

    // Process each image
    await Promise.all(images.map(async (base64Image, index) => {
      if (!/^data:image\/(jpg|jpeg|png);base64,/.test(base64Image)) {
        throw new Error(`Invalid image data format for image ${index}`);
      }

      const base64Data = base64Image.replace(/^data:image\/\w+;base64,/, '');
      const imgBuffer = Buffer.from(base64Data, 'base64');

      // Save the image with the index as the filename
      await sharp(imgBuffer)
        .jpeg({ quality: 90 })
        .toFile(path.join(dirPath, `${index}.jpeg`));
    }));

    // Load existing descriptors or initialize new object
    const jsonFilePath = path.join(__dirname, 'label_info/labeledImages.json');

    let dictionary = [];

    try {
      // Read existing descriptors
      const data = await fs_promise.readFile(jsonFilePath, 'utf8');
      dictionary = JSON.parse(data);
    } catch (error) {
      if (error.code !== 'ENOENT') throw error; // Ignore error if file does not exist, otherwise throw
    }

    // Append or update the new label and descriptors
    const labeledDescriptors = await loadLabeledImages([id]);

    dictionary.push(...labeledDescriptors);

    // Write the updated labeled data back to the JSON file
    await fs_promise.writeFile(jsonFilePath, JSON.stringify(dictionary, null, 2));

    console.log('JSON data saved successfully!');
    // Update or add new label info
    const data = await fs_promise.readFile(jsonFilePath, 'utf8');
    const descriptorsJson = JSON.parse(data);
    const labeledDescriptor = descriptorsJson.map(ld => {
      const descriptors = ld.descriptors.map(d => new Float32Array(d));
      return new faceapi.LabeledFaceDescriptors(ld.label, descriptors);
    });

    faceMatcher = new faceapi.FaceMatcher(labeledDescriptor, distanceThreshold);
    console.log('Loaded descriptors from JSON.');

    // Update or add new label info
    labelInfo[id] = { name, status, position }; // Update the global labelInfo object
    await updateLabelInfo(); // Write changes back to labelInfo.json

    return res.status(200).json({ message: 'Images processed and saved successfully', id });
  } catch (error) {
    console.error(`Error during processing: ${error.message}`);
    return res.status(500).json({ message: 'Internal Server Error during processing.' });
  }
}

async function updateLabelInfo() {
  const filePath = path.join(__dirname, 'label_info/labelInfo.json');
  try {
    const labelArray = Object.entries(labelInfo).map(([id, { name, status, position }]) => ({
      id, name, status, position
    }));
    await fs_promise.writeFile(filePath, JSON.stringify(labelArray, null, 2));
  } catch (error) {
    console.error('Failed to update label info:', error);
    throw error;
  }
}

async function handle_face_recognition(req, res) {
  try {
    const imageDataURL = req.body.image;

    if (!imageDataURL || !/^data:image\/(jpg|jpeg|png);base64,/.test(imageDataURL)) {
      return res.status(400).send('Invalid image data format.');
    }
    console.time('functionTimer');
    const imageTensor = await loadImageAndConvertToTensor(imageDataURL);
    console.timeEnd('functionTimer');
    console.time('functionTimer1');
    const result = await face_recog(imageTensor, faceMatcher); // gpu
    console.timeEnd('functionTimer1');

    imageTensor.dispose();
    if (result.name !== 'noface' || result.name !== 'unknown') {
      console.log('result',result)
      await saveDetectionToDatabase(result);
    }
    return res.json(result);

  } catch (error) {
    console.error(`Error during initialization: ${error.message}`);
    return res.status(500).send('Internal Server Error during initialization.');
  }
}

// .... Face Recognition .... //
async function face_recog(img, faceMatch) { // gpu
  try {
    console.time('functionTimer2');
    const faces = await faceapi
      .detectAllFaces(img, optionsSSDMobileNet)
      .withFaceLandmarks()
      .withFaceDescriptors(); // Detect faces in the image
    console.timeEnd('functionTimer2');

    if (faces.length > 0) {
      const matches = faces.map((d) => ({
        id: faceMatch.findBestMatch(d.descriptor)._label,
        descriptor: d.descriptor
      }));

      // Map matches to include status and position
      const results = matches.map(match => {
        const labelData = labelInfo[match.id] || { name: "unknown", status: "unknown", position: "unknown" };
        return {
          name: labelData.name,
          status: labelData.status,
          position: labelData.position
        };
      }).filter(match => match.name !== 'unknown');

      // Return the first matched result or unknown
      return results.length > 0 ? results[0] : { name: 'unknown', status: 'unknown', position: 'unknown' };
    }
    return { name: 'noface' };
  } catch (err) {
    console.error('Caught error', err.message);
    logger.error(`Error during server setup: ${err.message}`);
    return null;
  }
}

// .... Save to Database .... //
async function saveDetectionToDatabase(detection) {
  const { name, status, position } = detection;
  const time = new Date();

  // Create the image filename
  const imageFilename = `${name.replace(/\s+/g, '_')}.jpg`;

  // Check if the person was already recorded within the last 10 minutes
  const query = `
    SELECT * FROM moiapp_employee
    WHERE name = ? AND status = ? AND position = ? AND time >= NOW() - INTERVAL 10 MINUTE
    ORDER BY time DESC
    LIMIT 1
  `;

  db.query(query, [name, status, position], (err, results) => {
    if (err) {
      console.error('Error checking existing records:', err);
      return;
    }

    if (results.length === 0) {
      // Save the new detection to the database
      const insertQuery = `
        INSERT INTO moiapp_employee (name, status, position, time, image)
        VALUES (?, ?, ?, ?, ?)
      `;
      db.query(insertQuery, [name, status, position, time, imageFilename], (err, results) => {
        if (err) {
          console.error('Error inserting new record:', err);
          return;
        }
        console.log('Detection saved to database:', { name, status, position, time, imageFilename });
      });
    } else {
      console.log('Person already recorded within the last 10 minutes:', { name, status, position });
    }
  });
}



// .... Main function .... //
async function main() {
  try {
    await checkAndSetBackend();
    await initializeFaceAPI();
    await loadLabelInfo();

    const jsonFilePath = path.join(__dirname, 'label_info/labeledImages.json');
    const additionalDataPath = path.join(__dirname, 'label_info/labelInfo.json');
    const dirPath = path.join(__dirname, 'labeled_images');

    try {
      // Read existing descriptors
      const data = await fs_promise.readFile(jsonFilePath, 'utf8');
      let descriptorsJson = JSON.parse(data);

      // Read additional metadata
      const additionalData = await fs_promise.readFile(additionalDataPath, 'utf8');
      const additionalDataJson = JSON.parse(additionalData);

      // Read labels from the directory
      const existingLabels = await fs_promise.readdir(dirPath, 'utf-8');

      // Filter and merge descriptors with additional metadata
      descriptorsJson = descriptorsJson.filter(ld => existingLabels.includes(ld.label)).map(ld => {
        const additionalInfo = additionalDataJson.find(ad => ad.id === ld.label) || {};
        return {
          ...ld,
          name: additionalInfo.name || "unknown",
          status: additionalInfo.status || "unknown",
          position: additionalInfo.position || "unknown"
        };
      });

      // Filter descriptors whose labels are not present in the directory
      descriptorsJson = descriptorsJson.filter(ld => existingLabels.includes(ld.label));

      // Map descriptors to faceapi format
      const labeledDescriptors = descriptorsJson.map(ld => {
        const descriptors = ld.descriptors.map(d => new Float32Array(d));
        return new faceapi.LabeledFaceDescriptors(ld.label, descriptors);
      });

      // Update face matcher
      faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, distanceThreshold);
      console.log('Loaded descriptors from JSON.');

      // Save the filtered and updated descriptors back to the file
      const updatedJson = descriptorsJson.map(ld => ({
        label: ld.label,
        descriptors: ld.descriptors.map(d => Array.from(d)),
        name: ld.name,
        status: ld.status,
        position: ld.position
      }));

      await fs_promise.writeFile(jsonFilePath, JSON.stringify(updatedJson, null, 2));
      console.log('Updated JSON data saved successfully.');

    } catch (error) {
      if (error.code === 'ENOENT') {
        console.log('JSON file not found, creating new descriptors.');

        const labels = await fs_promise.readdir(dirPath);
        console.log('here', labels);
        const labeledDescriptors = await loadLabeledImages(labels);

        faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, distanceThreshold);
        // Include metadata in initial creation if available
        const additionalData = await fs_promise.readFile(additionalDataPath, 'utf8');
        const additionalDataJson = JSON.parse(additionalData);

        const descriptorsJson = labeledDescriptors.map(ld => {
          const additionalInfo = additionalDataJson.find(ad => ad.id === ld.label) || {};
          return {
            label: ld.label,
            descriptors: ld.descriptors.map(d => Array.from(d)),
            name: additionalInfo.name || "unknown",
            status: additionalInfo.status || "unknown",
            position: additionalInfo.position || "unknown"
          };
        });

        await fs_promise.writeFile(jsonFilePath, JSON.stringify(descriptorsJson, null, 2));
        console.log('JSON data saved successfully!');
      } else {
        throw error;
      }
    }

    setupServer(); // Setup the server

  } catch (error) {
    console.error('Error during server initialization:', error);
  }
}

main();
