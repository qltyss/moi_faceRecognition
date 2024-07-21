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
    app.post('/delete_face', cors(), delete_face);
    app.post('/edit_face', cors(), edit_face);

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

    // Check if the name already exists in the database
    const checkQuery = `SELECT COUNT(*) as count FROM moiapp_employee WHERE name = ?`;
    db.query(checkQuery, [name], async (err, result) => {
      if (err) {
        console.error('Error checking existing name:', err);
        return res.status(500).json({ message: 'Internal server error' });
      }

      if (result[0].count > 0) {
        return res.status(409).json({ message: 'Name already exists' });
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

      // Save new face information to the database
      const imageFilename = `${name.replace(/\s+/g, '_')}.jpg`;
      const query = `
        INSERT INTO moiapp_employee (id, name, position, status, image, time)
        VALUES (?, ?, ?, ?, ?, NOW())
      `;
      db.query(query, [id, name, position, status, imageFilename], (err, result) => {
        if (err) {
          console.error('Error saving new face to database:', err);
          return res.status(500).json({ message: 'Internal server error' });
        }

        console.log('New face information saved to database:', result);
        return res.status(200).json({ message: 'New face added successfully', id: id });
      });

    });

  } catch (error) {
    console.error('Error handling new face:', error);
    logger.error(`Error handling new face: ${error.message}`);
    res.status(500).send('Internal server error');
  }
}


async function handle_face_recognition(req, res) {
  try {
    const imageDataURL = req.body.image;

    if (!imageDataURL || !/^data:image\/(jpg|jpeg|png);base64,/.test(imageDataURL)) {
      return res.status(400).send('Invalid image data format.');
    }
    
    const imageTensor = await loadImageAndConvertToTensor(imageDataURL);
    
    const id = await face_recog(imageTensor, faceMatcher); // Get the ID from face recognition
    console.log(id)
    imageTensor.dispose();

    const result = { id }; // Construct the response with just the ID

    return res.json(result);

  } catch (error) {
    console.error(`Error during face recognition: ${error.message}`);
    return res.status(500).send('Internal Server Error during face recognition.');
  }
}

async function delete_face(req, res) {
  const id = req.body.id;
  
  try {
    // Check if the employee exists
    const checkQuery = `SELECT name FROM moiapp_employee WHERE id = ?`;
    db.query(checkQuery, [id], async (err, result) => {
      if (err) {
        console.error('Error checking existing employee:', err);
        return res.status(500).json({ message: 'Internal server error' });
      }

      if (result.length === 0) {
        return res.status(200).json({ message: 'Employee not found' });
      }

      const employeeName = result[0].name;
      console.log(employeeName)

      // Delete employee from the database
      const deleteQuery = `DELETE FROM moiapp_employee WHERE id = ?`;
      db.query(deleteQuery, [id], async (err, result) => {
        if (err) {
          console.error('Error deleting employee from database:', err);
          return res.status(500).json({ message: 'Internal server error' });
        }

        // Delete employee's images
        const dirPath = path.join(__dirname, 'labeled_images', id);
        await fs_promise.rm(dirPath, { recursive: true, force: true }).catch(err => {
          console.error('Error deleting employee images:', err);
          return res.status(500).json({ message: 'Internal server error' });
        });

        // Update the labeledImages.json file
        const jsonFilePath = path.join(__dirname, 'label_info/labeledImages.json');
        let dictionary = [];

        try {
          const data = await fs_promise.readFile(jsonFilePath, 'utf8');
          dictionary = JSON.parse(data);
        } catch (error) {
          if (error.code !== 'ENOENT') throw error;
        }

        dictionary = dictionary.filter(descriptor => descriptor.label !== id);

        await fs_promise.writeFile(jsonFilePath, JSON.stringify(dictionary, null, 2));

        console.log('Employee deleted successfully from JSON and file system');

        // Update faceMatcher with the new descriptors
        const descriptorsJson = JSON.parse(await fs_promise.readFile(jsonFilePath, 'utf8'));
        const labeledDescriptor = descriptorsJson.map(ld => {
          const descriptors = ld.descriptors.map(d => new Float32Array(d));
          return new faceapi.LabeledFaceDescriptors(ld.label, descriptors);
        });

        faceMatcher = new faceapi.FaceMatcher(labeledDescriptor, distanceThreshold);
        console.log('Updated descriptors from JSON.');

        return res.status(200).json({ message: 'Employee deleted successfully' });
      });
    });
  } catch (error) {
    console.error('Error handling employee deletion:', error);
    logger.error(`Error handling employee deletion: ${error.message}`);
    res.status(500).send('Internal server error');
  }
}

async function edit_face(req, res) {
  const { id, name, position, status } = req.body;

  if (!id || !name || !position || !status) {
    return res.status(200).json({message:'invalid input'});
  }

  try {
    // Check if the employee exists
    const checkQuery = `SELECT COUNT(*) as count FROM moiapp_employee WHERE id = ?`;
    db.query(checkQuery, [id], async (err, result) => {
      if (err) {
        console.error('Error checking existing employee:', err);
        return res.status(500).json({ message: 'Internal server error' });
      }

      if (result[0].count === 0) {
        return res.status(200).json({ message: 'Employee not found' });
      }

      // Check if the name already exists
      const nameCheckQuery = `SELECT COUNT(*) as count FROM moiapp_employee WHERE name = ? AND id != ?`;
      db.query(nameCheckQuery, [name, id], (err, result) => {
        if (err) {
          console.error('Error checking existing name:', err);
          return res.status(500).json({ message: 'Internal server error' });
        }

          if (result[0].count > 0) {
            return res.status(200).json({ message: 'Employee name already exists' });
          }

        // Set the current timestamp
        const currentTime = new Date();
        const imageFilename = `${name.replace(/\s+/g, '_')}.jpg`;
        
        // Update employee details in the database
        const updateQuery = `
          UPDATE moiapp_employee
          SET name = ?, position = ?, status = ?, image = ?, time = ?
          WHERE id = ?
        `;
        db.query(updateQuery, [name, position, status, imageFilename, currentTime, id], (err, result) => {
          if (err) {
            console.error('Error updating employee details:', err);
            return res.status(500).json({ message: 'Internal server error' });
          }

          console.log('Employee details updated successfully:', result);
          return res.status(200).json({ message: 'Employee details updated successfully' });
        });
      });
    });
  } catch (error) {
    console.error('Error handling employee update:', error);
    logger.error(`Error handling employee update: ${error.message}`);
    res.status(500).send('Internal server error');
  }
}


// .... Face Recognition .... //
async function face_recog(img, faceMatch) { // gpu
  try {
    
    const faces = await faceapi
      .detectAllFaces(img, optionsSSDMobileNet)
      .withFaceLandmarks()
      .withFaceDescriptors(); // Detect faces in the image
    

    if (faces.length > 0) {
      const matches = faces.map((d) => ({
        id: faceMatch.findBestMatch(d.descriptor)._label
      }));

      // Return the ID of the first matched result
      return matches.length > 0 ? matches[0].id : 'unknown';
    }
    return 'noface';
  } catch (err) {
    console.error('Caught error', err.message);
    logger.error(`Error during face recognition: ${err.message}`);
    return 'error';
  }
}


// .... Main function .... //
async function main() {
  try {
    await checkAndSetBackend();
    await initializeFaceAPI();
    

    const jsonFilePath = path.join(__dirname, 'label_info/labeledImages.json');
    const dirPath = path.join(__dirname, 'labeled_images');

    try {
      // Read existing descriptors
      const data = await fs_promise.readFile(jsonFilePath, 'utf8');
      let descriptorsJson = JSON.parse(data);

      // Read labels from the directory
      const existingLabels = await fs_promise.readdir(dirPath, 'utf-8');

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

      // Save the filtered descriptors back to the file (excluding metadata)
      const updatedJson = descriptorsJson.map(ld => ({
        label: ld.label,
        descriptors: ld.descriptors.map(d => Array.from(d)),
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

        // Save descriptors without metadata
        const descriptorsJson = labeledDescriptors.map(ld => ({
          label: ld.label,
          descriptors: ld.descriptors.map(d => Array.from(d)),
        }));

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
