from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional, List, Any
from pydantic import BaseModel
import os
import uuid
import logging
from fastapi.responses import JSONResponse
from datetime import datetime, timezone, timedelta
from app.livekit_client import LiveKitClient
from app.llm_client import LLMClient

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

livekit_client = LiveKitClient()
llm_client = LLMClient()

# In-memory storage for demo
active_rooms: Dict[str, Dict] = {}
call_summaries: Dict[str, str] = {}
transfer_sessions: Dict[str, Dict] = {}

class TokenRequest(BaseModel):
    room_name: str
    participant_name: str
    role: str = "participant"

class CreateRoomRequest(BaseModel):
    room_name: str
    max_participants: int = 5

class TransferRequest(BaseModel):
    caller_room: str
    agent_a_id: str
    agent_b_id: str
    call_context: Optional[str] = None

class SummaryRequest(BaseModel):
    conversation_history: List[Dict[str, Any]]
    caller_info: Optional[Dict[str, Any]] = None 

class CompleteTransferRequest(BaseModel):
    transfer_id: str
    caller_room: str
    new_room: str


@app.get("/")
async def root():
    return {
        "message": "I came, I saw",
        "status": "online",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/health")
async def health_check():
    try:
        # Test LiveKit connection
        rooms = await livekit_client.list_rooms()
        return {
            "status": "healthy",
            "livekit": "connected",
            "active_rooms": len(rooms),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )
    
# Authentication endpoints
@app.post("/api/token")
async def generate_token(request: TokenRequest):
    """
    Generate LiveKit access token for room participant
    """

    try:
        token = await livekit_client.generate_token(
            room_name=request.room_name,
            participant_name=request.participant_name,
            role=request.role
        )

        logging.info(f"Generated token for {request.participant_name} in room {request.room_name}")

        return {
            "token": token,
            "room_name": request.room_name,
            "participant_name": request.participant_name,
            "livekit_url": os.getenv("LIVEKIT_URL")
        }
    except Exception as e:
        logging.error(f"Token generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate token: {str(e)}")
    
# Room management endpoints
@app.post("/api/create-room")
async def create_room(request: CreateRoomRequest):
    """
    Create a new LiveKit room
    """

    try:
        room_info = await livekit_client.create_room(
            room_name=request.room_name,
            max_participants=request.max_participants
        )

        # Store room info
        active_rooms[request.room_name] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "max_participants": request.max_participants,
            "participants": [],
            "status": "active"
        }

        logging.info(f"Created room: {request.room_name}")

        return {
            "room_name": request.room_name,
            "room_info": room_info,
            "created_at": active_rooms[request.room_name]["created_at"]
        }
    except Exception as e:
        logging.error(f"Room creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create room: {str(e)}")
    
@app.get("/api/rooms")
async def list_rooms():
    """
    List all active rooms
    """

    try:
        livekit_rooms = await livekit_client.list_rooms()

        return {
            "active_rooms": active_rooms,
            "livekit_rooms": livekit_rooms,
            "total_rooms": len(active_rooms)
        }
    except Exception as e:
        logging.error(f"Failed to list rooms: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list rooms: {str(e)}")
    
@app.delete("/api/rooms/{room_name}")
async def delete_room(room_name: str):
    """
    Delete a room and clean up resources
    """

    try:
        await livekit_client.delete_room(room_name)

        # Clean up local storage
        if room_name in active_rooms:
            del active_rooms[room_name]

        logging.info(f"Deleted room: {room_name}")

        return {"message": f"Room {room_name} deleted successfully"}
    except Exception as e:
        logging.error(f"Room deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete room: {str(e)}")
    
# Transfer workflow endpoints
@app.post("/api/initiate-transfer")
async def initiate_transfer(request: TransferRequest):
    """
    Initiate warm transfer process
    """

    try:
        transfer_id = str(uuid.uuid4())
        transfer_room = f"transfer_{transfer_id}"

        # Create transfer room for Agent A and B
        await livekit_client.create_room(transfer_room, max_participants=2)

        # Store transfer session
        transfer_sessions[transfer_id] = {
            "caller_room": request.caller_room,
            "transfer_room": transfer_room,
            "agent_a_id": request.agent_a_id,
            "agent_b_id": request.agent_b_id,
            "status": "initiated",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "call_context": request.call_context
        }

        logging.info(f"Initiated transfer {transfer_id} from {request.caller_room}")

        return {
            "transfer_id": transfer_id,
            "transfer_room": transfer_room,
            "status": "initiated",
            "agent_a_token": await livekit_client.generate_token(
                transfer_room, request.agent_a_id, "agent"
            ),
            "agent_b_token": await livekit_client.generate_token(
                transfer_room, request.agent_b_id, "agent"
            )
        }
    except Exception as e:
        logging.error(f"Transfer intiation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate transfer: {str(e)}")
    
@app.post("/api/generate-summary")
async def generate_call_summary(request: SummaryRequest):
    """
    Generate AI-powered call summary
    """

    try:
        summary = await llm_client.generate_call_summary(
            conversation_history=request.conversation_history,
            caller_info=request.caller_info
        )

        # Store summary for later use
        summary_id = str(uuid.uuid4())
        call_summaries[summary_id] = summary

        logging.info(f"Generated call summary {summary_id}")

        return {
            "summary_id": summary_id,
            "summary": summary,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logging.error(f"Summary generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")
    
@app.post("/api/complete-transfer")
async def complete_transfer(request: CompleteTransferRequest):
    """
    Complete the warm transfer process
    """

    try:
        if request.transfer_id not in transfer_sessions:
            raise HTTPException(status_code=404, detail=f"Transfer session not found")
        
        transfer_session = transfer_sessions[request.transfer_id]

        # Update transfer status
        transfer_session["status"] = "completed"
        transfer_session["completed_at"] = datetime.now(timezone.utc).isoformat()

        # Generate token for Agent B to join caller room
        agent_b_token = await livekit_client.generate_token(
            request.caller_room,
            transfer_session["agent_b_id"],
            "agent"
        )

        logging.info(f"Completed transfer {request.transfer_id}")

        return {
            "transfer_id": request.transfer_id,
            "status": "completed",
            "agent_b_token": agent_b_token,
            "caller_room": request.caller_room
        }
    except Exception as e:
        logging.error(f"Transfer completion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to complete transfer: {str(e)}")
    
@app.get("/api/transfer/{transfer_id}")
async def get_transfer_status(transfer_id: str):
    """
    Get transfer session status
    """

    if transfer_id not in transfer_sessions:
        raise HTTPException(status_code=404, detail="Transfer session not found")
    
    return {
        "transfer_id": transfer_id,
        "session": transfer_sessions[transfer_id]
    }

# Utility endpoints
@app.get("/api/summaries/{summary_id}")
async def get_summary(summary_id: str):
    """
    Retrieve a stored call summary
    """

    if summary_id not in call_summaries:
        raise HTTPException(status_code=404, detail="Summary not found")
    
    return {
        "summary_id": summary_id,
        "summary": call_summaries[summary_id]
    }

@app.get("/api/stats")
async def get_stats():
    """
    Get system statistics
    """

    return {
        "active_rooms": len(active_rooms),
        "active_transfers": len([t for t in transfer_sessions.values() if t["status"] == "initiated"]),
        "completed_transfers": len([t for t in transfer_sessions.values() if t["status"] == "completed"]),
        "total_summaries": len(call_summaries),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# Background tasks for cleanup
async def cleanup_old_sessions():
    """
    Clean up old transfer sessions and summaries
    """

    current_time = datetime.now(timezone.utc)

    # Clean up transfer sessions older than 1 hour
    expired_transfers = []
    for transfer_id, session in transfer_sessions.items():
        created_at = datetime.fromisoformat(session["created_at"])
        if current_time - created_at > timedelta(hours=1):
            expired_transfers.append(transfer_id)

    for transfer_id in expired_transfers:
        del transfer_sessions[transfer_id]
        logging.info(f"Cleaned up expired transfer session: {transfer_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
        


