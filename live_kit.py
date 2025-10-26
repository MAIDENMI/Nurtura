import asyncio
import logging
import base64
import cv2
import numpy as np

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    metrics,
    function_tool,
    RunContext,
)
from livekit.agents.llm import ImageContent
from livekit.agents.utils.images import encode, EncodeOptions, ResizeOptions
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self, vital_summary: str = None, video_frames: list = None) -> None:
        self.base_instructions = """You are Halo, a robotic baby monitoring system. You report only abnormal events. Tone: clinical and brief.

SPEAK-ONLY RULE:
- Speak only for abnormal vitals, safety hazards, or acoustic/visual events. Otherwise, stay silent.

Inputs:
1) Vital-signs analysis (HR/BR trends).
2) Video frames (posture, movement, crib boundary).
3) Acoustic events (non-speech), e.g., baby_cry with timestamps.

Priority:
1) Egress risk (climbing/at edge/out of crib)
2) Airway risk (prone face-down, obstructed airway, apnea-like stillness)
3) Vital sign abnormalities (HR/BR out of range, irregularity clusters)
4) Cry distress
5) Temperature anomalies (if provided)

Cry rules (no confidence used):
- Alert if baby_cry lasts ≥ 5s continuously, OR
- ≥ 3 baby_cry bursts occur within 20s.
- If crying overlaps with egress/airway risk, escalate.
- After crying stops for ≥ 30s, suppress further alerts until rules are met again.

Reporting (one short sentence):
- “Alert: crying detected. Posture: supine. Vitals: within range.”
- If multiple issues, mention the highest-priority one first.

If no alert criteria are met: remain silent.      
"""

        if vital_summary:
            instructions = f"{self.base_instructions}\n\nCurrent vital signs summary: {vital_summary}"
        else:
            instructions = self.base_instructions
        
        self.video_frames = video_frames or []
            
        super().__init__(instructions=instructions)

    @function_tool
    async def end_conversation(self, context: RunContext):
        logger.info("User requested to end conversation")
        raise Exception("User requested to end conversation")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext, vital_summary: str = None, video_frames: list = None):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        user_away_timeout=0.0,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    assistant = Assistant(vital_summary=vital_summary, video_frames=video_frames)

    if video_frames and len(video_frames) > 0:
        content_items = ["Here are video frames from the monitoring camera:"]
        frame_indices = np.linspace(0, len(video_frames) - 1, min(5, len(video_frames)), dtype=int)
        
        for idx in frame_indices:
            frame = video_frames[idx]
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            
            content_items.append(
                ImageContent(
                    image=f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}",
                )
            )
        initial_ctx = ChatContext()
        initial_ctx.add_message(
            role="user",
            content=content_items
        )
        await assistant.update_chat_ctx(initial_ctx)
        await session.start(
            agent=assistant,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )
        await session.generate_reply(
            instructions="Analyze the vital signs summary and video frames if provided. If there are any abnormalities or concerns, report them briefly and clinically. If everything is normal, remain silent and generate no audio output.",
            allow_interruptions=False,
        )
        ctx.shutdown()
        print("Session generated reply")


def call_live_kit(message, video_frames=None):
    async def entrypoint_with_sample(ctx: JobContext):
        return await entrypoint(ctx, message, video_frames)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint_with_sample, prewarm_fnc=prewarm))
