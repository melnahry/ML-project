const express = require('express');
const multer = require('multer');
const request = require('request');
const fs = require('fs');

const app = express();
const port = 3001;

// Set up Multer for handling file uploads
const upload = multer({ dest: 'uploads/' });

// Serve static files from the 'public' directory
app.use(express.static('public'));

// Define a route for handling file uploads
app.post('/upload', upload.single('file'), (req, res) => {
    // Construct the URL for the Flask server
    const flaskURL = 'http://localhost:3000';

    // Read the uploaded file from the disk
    const fileData = fs.readFileSync(req.file.path);

    // Create a FormData object to send the file
    const formData = {
        file: {
            value: fileData,
            options: {
                filename: req.file.originalname
            }
        }
    };

    // Make a POST request to the Flask server
    request.post({ url: flaskURL, formData: formData }, (error, response, body) => {
        if (error) {
            console.error(error);
            res.status(500).send('Internal Server Error');
        } else {
            res.send(body);
        }
    });
});

// Define a route handler for the root ("/") URL path
app.get('/', (req, res) => {
    // Here you can serve your HTML file or render a template
    res.sendFile(__dirname + '/index.html');
});

// Start the server
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
