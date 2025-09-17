### Warm Transfer Backend (LiveKit + LLM)

Backend implementation of a **Warm Transfer** system using **LiveKit** and **LLMs**.  
This service handles token generation, room creation, call summaries, and the warm transfer flow between agents.  
*(Frontend not implemented yet.)*

---

**Features**

- Generate LiveKit access tokens for clients  
- Create and list LiveKit rooms  
- Initiate warm transfer from one agent to another  
- Generate AI-powered call summaries via OpenAI or Groq  
- Complete transfer by handing off caller to new agent  

---

### 🛠️ Setup  

### 1. Clone Repository  

```bash
git clone https://github.com/your-username/attack-capital-warm-transfer
cd attack-capital-warm-transfer
```

### 2. Backend setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
```

### 4. Edit .env with your own keys:

```bash
# LiveKit Configuration
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
LIVEKIT_URL=wss://your-project.livekit.cloud

# LLM Configuration (choose one)
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
LLM_PROVIDER=openai  # or 'groq'
```

---

### API Endpoints  

| Method | Endpoint                 | Description                     |
|--------|--------------------------|---------------------------------|
| **POST** | `/api/token`             | Generate LiveKit access token   |
| **POST** | `/api/create-room`       | Create new LiveKit room         |
| **GET**  | `/api/rooms`             | List active rooms               |
| **POST** | `/api/initiate-transfer` | Start warm transfer process     |
| **POST** | `/api/generate-summary`  | Create AI call summary          |
| **POST** | `/api/complete-transfer` | Finalize agent handoff          |


