import asyncio
import json
import os
from typing import Optional

from livekit import api, rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    Agent,
    AgentSession,
    cli,
    llm,
    stt,
    tts,
    vad,
)
from livekit.plugins import baseten, silero, openai


config = {
    "baseten_model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "baseten_api_key": "APItGErLWJegDVo",
    "phone_number": "+19037876024",
    "alert_message": "This is an automated alert message. Please respond to confirm receipt."
}


class PhoneCallAlertAgent(Agent):
    """LiveKit Agent for making automated phone call alerts"""
    
    def __init__(self, phone_number: str, alert_message: str):
        # Initialize the base Agent class
        super().__init__(
            instructions=f"""You are an automated phone call alert system. 
            Your task is to deliver an important alert message: "{alert_message}"
            
            Guidelines:
            - Greet the recipient professionally
            - Deliver the alert message clearly
            - Ask for confirmation that they received the message
            - Thank them and end the call politely
            - Keep the conversation brief and professional
            - The recipient may be busy or in an emergency situation
            """,
            llm=baseten.LLM(
                model=config.baseten_model,
                api_key=config.baseten_api_key
            ),
            tts=silero.TTS(),
            stt=silero.STT(),
            vad=vad.VAD.create(),
        )
        
        self.phone_number = phone_number
        self.alert_message = alert_message


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit agent"""
    if not config.validate():
        print("Configuration validation failed")
        return
    
    # Parse job metadata to get phone number and alert message
    try:
        metadata = json.loads(ctx.job.metadata) if ctx.job.metadata else {}
        phone_number = metadata.get("phone_number", config.phone_number)
        alert_message = metadata.get("alert_message", config.alert_message)
    except (json.JSONDecodeError, AttributeError):
        phone_number = config.phone_number
        alert_message = config.alert_message
    
    # Create the agent
    agent = PhoneCallAlertAgent(phone_number, alert_message)
    
    # Create and start the agent session
    session = AgentSession(ctx, agent)
    
    # Handle outbound calls
    if phone_number and phone_number != config.phone_number:
        await handle_outbound_call(ctx, phone_number)
    
    # Start the session
    await session.start()
    
    # Generate initial greeting for inbound calls
    if not phone_number or phone_number == config.phone_number:
        await session.generate_reply(
            instructions="Greet the caller and deliver the alert message professionally."
        )


async def handle_outbound_call(ctx: JobContext, phone_number: str):
    """Handle outbound phone call creation"""
    try:
        # Create SIP participant for outbound call
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=config.sip_trunk_id,  # You'll need to set this in config
                sip_call_to=phone_number,
                participant_identity=phone_number,
                wait_until_answered=True,
            )
        )
        print(f"Successfully placed call to {phone_number}")
        
    except api.TwirpError as e:
        print(f"Error creating SIP participant: {e.message}")
        print(f"SIP status: {e.metadata.get('sip_status_code')} {e.metadata.get('sip_status')}")
        ctx.shutdown("Failed to place outbound call")
    except Exception as e:
        print(f"Unexpected error placing call: {e}")
        ctx.shutdown("Failed to place outbound call")


async def make_standalone_call(phone_number: str, alert_message: str):
    """Make a standalone outbound call using LiveKit dispatch"""
    try:
        # This would require setting up a LiveKit API client
        # and creating a dispatch request
        print(f"Would place call to {phone_number} with message: {alert_message}")
        return True
    except Exception as e:
        print(f"Error making standalone call: {e}")
        return False


if __name__ == "__main__":
    # Run the agent with explicit dispatch
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="phone-call-alert-agent"
        )
    )
