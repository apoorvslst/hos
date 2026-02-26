const path = require('path');
require('dotenv').config({ path: path.resolve(__dirname, '..', '.env') });
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const authRoutes = require('./routes/auth');
const voiceRoutes = require('./routes/voice');
const inboundRoutes = require('./routes/inbound');
const dashboardRoutes = require('./routes/dashboard');
const { initSummaryCron } = require('./services/summaryCron');

const app = express();

// Middleware
const allowedOrigins = process.env.ALLOWED_ORIGINS
    ? process.env.ALLOWED_ORIGINS.split(',').map(o => o.trim().toLowerCase())
    : ['http://localhost:5173'];

const corsOptions = {
    origin: function (origin, callback) {
        // Log the incoming origin for debugging in Railway logs
        console.log(`[CORS Check] Incoming Origin: ${origin}`);

        // allow requests with no origin (like mobile apps or curl requests)
        if (!origin) return callback(null, true);

        const normalizedOrigin = origin.toLowerCase().trim();

        const isAllowed = allowedOrigins.some(allowed => {
            if (allowed.includes('*')) {
                const regex = new RegExp('^' + allowed.replace(/\./g, '\\.').replace(/\*/g, '.*') + '$');
                return regex.test(normalizedOrigin);
            }
            return allowed === normalizedOrigin;
        });

        // Also allow Vercel subdomains for the project automatically
        const isVercelPreview = normalizedOrigin.endsWith('.vercel.app');

        if (isAllowed || isVercelPreview) {
            callback(null, true);
        } else {
            console.warn(`CORS BLOCKED for origin: ${origin}`);
            console.warn(`Allowed origins are: ${allowedOrigins.join(', ')}`);
            callback(null, false); // Return false instead of Error to avoid crashing in some express setups
        }
    },
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With', 'Accept'],
    credentials: true,
    preflightContinue: false,
    optionsSuccessStatus: 204
};

app.use(cors(corsOptions));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use('/public', express.static('public'));

// Request Logger for debugging
app.use((req, res, next) => {
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.path} | Origin: ${req.get('origin') || 'no-origin'}`);
    next();
});

// Health Checks
app.get('/health', (req, res) => res.json({ status: 'ok', service: 'node-backend', time: new Date() }));

const axios = require('axios');

// Proxy Health Check (Verify if Python is reachable)
app.get('/api/ai-health', async (req, res) => {
    try {
        const pythonRes = await axios.get('http://localhost:8000/docs'); // FastAPI docs available by default
        res.json({ status: 'ok', service: 'python-ai', python_status: pythonRes.status });
    } catch (error) {
        res.status(500).json({ status: 'error', message: 'Python service unreachable', error: error.message });
    }
});

// Proxy route for /api/ask (Python AI service)
app.post('/api/ask', async (req, res) => {
    try {
        console.log(`[AI Proxy] Forwarding to Python: /ask`);
        const pythonResponse = await axios.post('http://localhost:8000/ask', req.body, {
            headers: { 'Content-Type': 'application/json' }
        });
        res.json(pythonResponse.data);
    } catch (error) {
        console.error('Error proxying to Python service:', error.message || error.code || 'Unknown error');
        console.error('  → Is Python running on port 8000?', error.code === 'ECONNREFUSED' ? 'NO - Python service is not reachable' : '');
        res.status(error.response?.status || 502).json({
            message: 'AI Service Error - Python RAG service may still be starting up',
            error: error.message || error.code,
            details: error.response?.data
        });
    }
});

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/voice', voiceRoutes);
app.use('/api/inbound', inboundRoutes);
app.use('/api/dashboard', dashboardRoutes);

// Database Connection + Cron Initialization
const connectDB = async () => {
    try {
        await mongoose.connect(process.env.MONGO_URI);
        console.log('MongoDB Connected successfully');

        // Start summary cron jobs after DB is ready
        initSummaryCron();
    } catch (err) {
        console.error('MongoDB connection error:', err.message);
        process.exit(1);
    }
};

connectDB();

// Global Error Handler
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ message: 'Internal Server Error', error: err.message });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});

