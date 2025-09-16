import os
import jwt
import logging
from typing import Dict, List, Optional, Any
from livekit import api
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveKitClient:
    """
    LiveKit client wrapper for room and participant management
    """

    def __init__(self):
        self.api_key = os.getenv("LIVEKIT_API_KEY")
        self.api_secret = os.getenv("LIVEKIT_API_SECRET")
        self.livekit_url = os.getenv("LIVEKIT_URL")

        if not all([self.api_key, self.api_secret, self.livekit_url]):
            raise ValueError("Missing LiveKit credentials. Check LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL")
        
        # Initialize the LiveKit API client
        self.livekit_api = api.LiveKitAPI(
            url=self.livekit_url,
            api_key=self.api_key,
            api_secret=self.api_secret
        )
        
        # Access room service through the API client
        self.room_service = self.livekit_api.room
        
        logger.info("LiveKit client initialized successfully")

    async def generate_token(
        self,
        room_name: str,
        participant_name: str,
        role: str = "participant",
        duration_hours: int = 2
    ) -> str:
        """
        Generate JWT access token for LiveKit room access

        Args:
            room_name: Name of room to join
            participant_name: Unique participant identifier
            role: Role type (caller, agent_a, agent_b, participant)
            duration_hours: Token validity duration in hours

        Returns:
            JWT token string
        """

        try:
            permissions = self._get_permissions_for_role(role)

            access_token = api.AccessToken(self.api_key, self.api_secret)
            access_token.with_identity(participant_name)
            access_token.with_name(participant_name)
            access_token.with_grants(api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=permissions.get("can_publish", True),
                can_subscribe=permissions.get("can_subscribe", True),
                can_publish_data=permissions.get("can_publish_data", True)
            ))

            # Set token expiration
            access_token.with_ttl(timedelta(hours=duration_hours))

            token = access_token.to_jwt()

            logger.info(f"Generated token for {participant_name} in room {room_name} with role {role}")
            return token
        
        except Exception as e:
            logger.error(f"Failed to generate token: {str(e)}")
            raise Exception(f"Token generation failed: {str(e)}")
        
    def _get_permissions_for_role(self, role: str) -> Dict[str, bool]:
        """
        Get permissions based on participant role
        """

        role_permissions = {
            "caller": {
                "can_publish": True,
                "can_subscribe": True,
                "can_publish_data": False,
                "can_update_metadata": False
            },
            "agent_a": {
                "can_publish": True,
                "can_subscribe": True,
                "can_publish_data": True,
                "can_update_metadata": True
            },
            "agent_b": {
                "can_publish": True,
                "can_subscribe": True,
                "can_publish_data": True,
                "can_update_metadata": True
            },
            "participant": {
                "can_publish": True,
                "can_subscribe": True,
                "can_publish_data": False,
                "can_update_metadata": False
            }
        }

        return role_permissions.get(role, role_permissions["participant"])
    
    async def create_room(
        self,
        room_name: str,
        max_participants: int = 10,
        empty_timeout: int = 300,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new LiveKit room

        Args:
            room_name: Unique room identifier
            max_participants: Maximum number of participants
            empty_timeout: Seconds before empty room is deleted
            metadata: Additional room metadata

        Returns:
            Room creation response
        """

        try:
            room_info = await self.room_service.create_room(
                api.proto_room.CreateRoomRequest(
                    name=room_name,
                    max_participants=max_participants,
                    empty_timeout=empty_timeout,
                    metadata=str(metadata) if metadata else ""
                )
            )

            logger.info(f"Created room: {room_name} with max participants: {max_participants}")

            return {
                "sid": room_info.sid,
                "name": room_info.name,
                "max_participants": room_info.max_participants,
                "creation_time": room_info.creation_time,
                "turn_password": room_info.turn_password,
                "enabled_codecs": [codec.mime for codec in room_info.enabled_codecs],
                "metadata": room_info.metadata
            }
        
        except Exception as e:
            logger.error(f"Failed to create room {room_name}: {str(e)}")
            raise Exception(f"Room creation failed: {str(e)}")
        
    async def list_rooms(self) -> List[Dict[str, Any]]:
        """
        List all active rooms

        Returns:
            List of room information dictionaries
        """

        try:
            rooms_response = await self.room_service.list_rooms(api.proto_room.ListRoomsRequest())

            rooms_info = []
            for room in rooms_response.rooms:
                room_info = {
                    "sid": room.sid,
                    "name": room.name,
                    "max_participants": room.max_participants,
                    "num_participants": room.num_participants,
                    "num_publishers": room.num_publishers,
                    "creation_time": room.creation_time,
                    "metadata": room.metadata
                }
                rooms_info.append(room_info)

            logger.info(f"Listed {len(rooms_info)} active rooms")
            return rooms_info
        
        except Exception as e:
            logger.error(f"Failed to list rooms: {str(e)}")
            raise Exception(f"Room listing failed: {str(e)}")
        
    async def get_room(self, room_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific room

        Args:
            room_name: name of the room to query

        Returns:
            Room information or None if not found
        """

        try:
            rooms = await self.list_rooms()
            for room in rooms:
                if room["name"] == room_name:
                    return room
            return None
        
        except Exception as e:
            logger.error(f"Failed to get room {room_name}: {str(e)}")
            return None
        
    async def delete_room(self, room_name: str) -> bool:
        """
        Delete a room and disconnect all participants

        Args:
            room_name: Name of the room to delete

        Returns:
            True if successful, False otherwise
        """

        try:
            await self.room_service.delete_room(
                api.proto_room.DeleteRoomRequest(room=room_name)
            )

            logger.info(f"Deleted room: {room_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete room {room_name}: {str(e)}")
            return False
        
    async def list_participants(self, room_name: str) -> List[Dict[str, Any]]:
        """
        List all participants in a room

        Args:
            room_name: Name of the room

        Returns: 
            List of participant information
        """

        try:
            participants_response = await self.room_service.list_participants(
                api.proto_room.ListParticipantsRequest(room=room_name)
            )

            participants_info = []
            for participant in participants_response.participants:
                participant_info = {
                    "sid": participant.sid,
                    "identity": participant.identity,
                    "name": participant.name,
                    "state": participant.state,
                    "joined_at": participant.joined_at,
                    "is_publisher": participant.is_publisher,
                    "metadata": participant.metadata,
                    "tracks": [
                        {
                            "sid": track.sid,
                            "name": track.name,
                            "type": track.type,
                            "source": track.source,
                            "muted": track.muted
                        }
                        for track in participant.tracks
                    ]
                }
                participants_info.append(participant_info)

            logger.info(f"Listed {len(participants_info)} participants in room {room_name}")
            return participants_info
        
        except Exception as e:
            logger.error(f"Failed to list participants in room {room_name}: {str(e)}")
            return []
        
    async def remove_participant(self, room_name: str, participant_identity: str) -> bool:
        """
        Remove a participant from a room

        Args:
            room_name: Name of the room
            participant_identity: Identity of participant to remove

        Returns:
            True if successful, False otherwise
        """

        try:
            await self.room_service.remove_participant(
                api.proto_room.RoomParticipantIdentity(
                    room=room_name,
                    identity=participant_identity
                )
            )

            logger.info(f"Removed participant {participant_identity} from room {room_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to remove participant {participant_identity} from room {room_name}: {str(e)}")
            return False
        
    async def update_participant_metadata(
        self,
        room_name: str,
        participant_identity: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Update participant metadata

        Args:
            room_name: Name of the room
            participant_identity: Identity of participant
            metadata: New metadata dictionary

        Returns:
            True if successful, False otherwise
        """

        try:
            await self.room_service.update_participant(
                api.proto_room.UpdateParticipantRequest(
                    room=room_name,
                    identity=participant_identity,
                    metadata=str(metadata)
                )
            )

            logger.info(f"Updated metadata for participant {participant_identity} in room {room_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update participant metadata: {str(e)}")
            return False
        
    async def send_data_message(
        self,
        room_name: str,
        data: str,
        destination_identities: Optional[List[str]] = None
    ) -> bool:
        """
        Send data message to participants in room

        Args:
            room_name: Name of the room
            data: Message data to send
            destination_identities: Specific participants to send to (None = all)

        Returns:
            True if successful, False otherwise
        """

        try:
            await self.room_service.send_data(
                api.proto_room.SendDataRequest(
                    room=room_name,
                    data=data.encode("utf-8"),
                    destination_identities=destination_identities or []
                )
            )

            logger.info(f"Sent data message to room {room_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send data message: {str(e)}")
            return False
        
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate and decode LiveKit JWT token

        Args:
            token: JWT token to validate

        Returns:
            Decoded token payload or None if invalid
        """

        try:
            payload = jwt.decode(
                token,
                self.api_secret,
                algorithms=["HS256"],
                options={"verify_exp": True}
            )

            logger.info(f"Token validated for identity: {payload.get('sub')}")
            return payload
        
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            return None
        
    async def get_room_stats(self, room_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed statistics for a room

        Args:
            room_name: Name of the room

        Returns:
            Room statistics dictionary
        """

        try:
            room = await self.get_room(room_name)
            if not room:
                return None
            
            participants = await self.list_participants(room_name)

            stats = {
                "room_info": room,
                "participant_count": len(participants),
                "publisher_count": sum(1 for p in participants if p["is_publisher"]),
                "subscriber_count": len(participants) - sum(1 for p in participants if p["is_publisher"]),
                "active_tracks": sum(len(p["tracks"]) for p in participants),
                "participants": participants
            }

            return stats
        
        except Exception as e:
            logger.error(f"Failed to get room stats for {room_name}: {str(e)}")
            return None

    async def close(self):
        """
        Close the LiveKit API client
        """
        await self.livekit_api.aclose()