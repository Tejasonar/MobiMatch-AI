MobiMatch-AI 📱

An intelligent smartphone recommendation system that uses a weighted scoring algorithm to find the best devices based on user intent.

 🚀 Overview
MobiMatch-AI goes beyond simple filtering. It evaluates technical specifications using a **Weighted Scoring Engine** to rank the best phones for specific use cases like Gaming, Photography, or General Balanced use.

✨ Key Features
* **Intent-Based Ranking:** Choose between 'Gaming', 'Photography', or 'Balanced' to change how specs are prioritized.
* **Data Normalization:** Uses `MinMaxScaler` to compare different units (GHz, mAh, GB) on a uniform scale.
* **Smart Filtering:** Real-time filtering by budget and Operating System (Android/iOS).
* **Interactive UI:** Built with Streamlit for a seamless, responsive user experience.

 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, Scikit-Learn, NumPy
* **Web Framework:** Streamlit
* **Deployment:** Streamlit Cloud

 📋 File Structure
* `app.py`: The main application code and recommendation logic.
* `requirements.txt`: List of Python dependencies.
* `mobile.csv`: The dataset containing smartphone specifications.

⚙️ Installation & Usage
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

 
