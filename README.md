**Possible Structure:**

**Input:** Cleaning Request\
        ↓\
LLM / NLP Processing (Extract features)\
        ↓\
Cleaner Profiles (Formed from dataset and mock data)\
        ↓\
Matching Algorithm (Regression model)\
        ↓\
**Output:** Best Cleaner Recommendation

**What needs to be built/setup:**
1. Front End UI
  - Possibly can make use of Streamlit or Flask for a basic application (both work great with Python)
  - User inputs: sq footage, cleaning type (deep, normal, etc.), any extra notes (free text)
2. AI Model
  - Use a LLM or simple model for extracting features from text 
  - Regression model for matching a cleaner to a job
3. Cleaner dataset of real and generated data
