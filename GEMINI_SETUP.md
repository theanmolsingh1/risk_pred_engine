# Gemini AI Setup Guide

## Getting Your Gemini API Key

1. **Visit Google AI Studio**: Go to https://makersuite.google.com/app/apikey
2. **Sign in** with your Google account
3. **Create API Key**: Click "Create API Key" button
4. **Copy the key**: Save your API key securely

## Setting Up the API Key

### Option 1: Environment Variable (Recommended)
```bash
# Windows PowerShell
$env:GEMINI_API_KEY="AIzaSyBVMv7C1qNBginv74z3CLP_KF5YIwNPeNo"

# Windows Command Prompt
set GEMINI_API_KEY=AIzaSyBVMv7C1qNBginv74z3CLP_KF5YIwNPeNo

# Then run your Flask app
cd backend
python app.py
```

### Option 2: Create .env file
1. Create a `.env` file in your project root
2. Add: `GEMINI_API_KEY=your_api_key_here`
3. Install python-dotenv: `pip install python-dotenv`
4. Update backend/app.py to load from .env file

## Features Added

### 🤖 AI Health Assistant
- **Local Page**: AI explains individual patient analysis
- **Global Page**: AI provides population health insights
- **Smart Explanations**: User-friendly medical explanations
- **Refresh Button**: Get updated AI analysis

### 📊 What AI Explains
- **Risk Score Meaning**: What 0-1 scale represents
- **Trend Analysis**: Whether patient is improving/worsening
- **Recommendations**: When to seek medical attention
- **Population Insights**: Overall health trends

## Usage
1. Set your API key using one of the methods above
2. Start the Flask server: `python app.py`
3. Upload patient data or view global database
4. AI explanations will appear automatically
5. Use "🔄 Refresh AI Analysis" for updated insights

## Troubleshooting
- **"AI explanations not available"**: Check if API key is set correctly
- **API errors**: Verify your API key is valid and has quota remaining
- **No response**: Check internet connection and API key permissions
