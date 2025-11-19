const express = require('express');
const axios = require('axios');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = 3000;

app.use(cors());

app.use(express.json());

app.use(express.static(path.join(__dirname)));

const PYTHON_API_URL = 'https://urban-health-prediction-1.onrender.com';

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.post('/get-prediction', async (req, res) => {
  try {
    const treeData = req.body;

    console.log('Received data from frontend:', treeData);

    if (!treeData || Object.keys(treeData).length === 0) {
      return res.status(400).json({ error: 'No data provided in request body.' });
    }

    console.log(`Sending data to Python ML service at ${PYTHON_API_URL}...`);

    const response = await axios.post(PYTHON_API_URL, treeData, { timeout: 10000 });

    const prediction = response.data;
    console.log('Received prediction from Python:', prediction);

    res.json(prediction);

  } catch (error) {
    console.error('Error in /get-prediction:', error.message);
    
    if (error.response) {
        console.error('Python server error:', error.response.data);
        res.status(500).json({ 
          error: 'Failed to get prediction from Python service.',
          details: error.response.data
        });
    } else {
        res.status(500).json({ 
          error: 'Failed to communicate with Python service.',
          details: error.message 
        });
    }
  }
});

app.listen(PORT, () => {
  console.log(`Node.js server listening at http://127.0.0.1:${PORT}`);
  console.log(`Frontend available at http://127.0.0.1:${PORT}`);
  console.log(`Python API expected at ${PYTHON_API_URL}`);
});
