# Crime Rate Prediction and Safety Analysis

<div align="center">
  <h2>🚨 Real-time Crime Risk Assessment System 🚨</h2>
  <p>A machine learning-powered application to analyze and predict crime risk levels across India</p>
</div>

## 🌟 Features

- **Real-time Risk Assessment**: Get instant safety scores for any location
- **Time-based Analysis**: Risk levels adjusted based on time of day
- **District & State Level Data**: Detailed analysis at both district and state levels
- **Smart Recommendations**: Contextual safety tips based on location and time
- **Interactive Map Interface**: Visual representation of safety levels

## 🚀 Getting Started

### Prerequisites

- Node.js (v14 or higher)
- Python 3.8+
- Git

### Installation

<details>
<summary>Frontend Setup</summary>

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```
- Access the application at [http://localhost:3000](http://localhost:3000)
</details>

<details>
<summary>Backend Setup</summary>

```bash
# Navigate to backend directory
cd backend



# Start the server
uvicorn app:app --reload
```
- API will be available at [http://localhost:8000](http://localhost:8000)
</details>

## 🔧 Available Scripts

### Frontend Commands

| Command | Description |
|---------|-------------|
| `npm start` | Runs development server |
| `npm test` | Launches test runner |
| `npm run build` | Builds for production |
| `npm run eject` | Ejects CRA configuration |

### Backend Commands

| Command | Description |
|---------|-------------|
| `uvicorn app:app --reload` | Starts development server |
| `python -m pytest` | Runs tests |

## 📚 Technical Details

### Frontend
- Built with React.js
- Material-UI for components
- Leaflet for maps
- Axios for API calls

### Backend
- FastAPI framework
- Scikit-learn for ML models
- Pandas for data processing
- Crime data analysis with customized algorithms

## 💡 Key Features Explained

### Risk Assessment Algorithm
- Analyzes historical crime data
- Considers multiple crime types with weighted severity
- Adjusts for population density
- Time-based risk modulation
- District and state-level comparisons

### Safety Recommendations
- Context-aware suggestions
- Time-based safety tips
- Location-specific precautions







## 🙏 Acknowledgments

- National Crime Records Bureau (NCRB) for data
- React team for Create React App
- FastAPI team for the backend framework


<div align="center">
  <p>Made with ❤️ for a safer tomorrow</p>
</div>
